#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstddef>
#include <set>
#include <variant>
#include <vector>

#include "llvm/ADT/BreadthFirstIterator.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"

#include "mlir/IR/BlockSupport.h"

namespace revng::detail {

template <class NodeT>
using StatusMap = llvm::DenseMap<NodeT, bool>;

template <class NodeT>
using EdgeDescriptor = std::pair<NodeT, NodeT>;

template <class NodeT>
class DFState : public StatusMap<NodeT> {
protected:
  using StatusMap = StatusMap<NodeT>;
  using EdgeDescriptor = EdgeDescriptor<NodeT>;

public:
  // Return the insertion iterator on the underlying map.
  std::pair<typename StatusMap::iterator, bool> insertInMap(NodeT Block,
                                                            bool OnStack) {
    return StatusMap::insert(std::make_pair(Block, OnStack));
  }

  // Return true if b is currently on the active stack of visit.
  bool onStack(NodeT Block) {
    auto Iter = this->find(Block);
    return Iter != this->end() && Iter->second;
  }

public:
  // Invoked after we have processed all children of a node during the DFS.
  void completed(NodeT Block) { (*this)[Block] = false; }
};

template <class NodeT>
class DFSBackedgeState : public DFState<NodeT> {
  using typename DFState<NodeT>::StatusMap;
  using typename DFState<NodeT>::EdgeDescriptor;

private:
  NodeT CurrentNode = nullptr;
  llvm::SmallSet<EdgeDescriptor, 4> Backedges;
  const std::function<bool(const NodeT)> &IsValid;

public:
  DFSBackedgeState(const std::function<bool(const NodeT)> &IsValid)
      : IsValid(IsValid) {}

  void setCurrentNode(NodeT Node) { CurrentNode = Node; }
  void insertBackedge(NodeT Source, NodeT Target) {
    Backedges.insert(std::make_pair(Source, Target));
  }
  llvm::SmallSet<EdgeDescriptor, 4> getBackedges() { return Backedges; }
  std::pair<typename StatusMap::iterator, bool> insert(NodeT Block) {

    // If we are trying to insert an invalid block, add it as not on the stack
    // so it does not influence the visit.
    if (IsValid(Block)) {
      return DFState<NodeT>::insertInMap(Block, true);
    } else {

      // We want to completely ignore the fact that we inserted an element in
      // the `DFState`, otherwise we will explore it anyway, therefore we
      // manually return `false`, so the node is not explored at all.
      auto InsertIt = DFState<NodeT>::insertInMap(Block, false);
      return std::make_pair(InsertIt.first, false);
    }
  }
};

template <class NodeT>
class DFSReachableState : public DFState<NodeT> {
  using typename DFState<NodeT>::StatusMap;
  using typename DFState<NodeT>::EdgeDescriptor;

private:
  // Set which contains the desired targets nodes marked as reachable during
  // the visit.
  llvm::SmallPtrSet<NodeT, 4> Targets;
  llvm::DenseMap<NodeT, llvm::SmallPtrSet<NodeT, 4>> AdditionalNodes;
  NodeT Source = nullptr;
  NodeT Target = nullptr;
  bool FirstInvocation = true;

public:
  // Insert the initial target node at the beginning of the visit.
  void insertTarget(NodeT Block) { Targets.insert(Block); }

  // Assign the `Source` node.
  void assignSource(NodeT Block) { Source = Block; }

  // Assign the `Target` node.
  void assignTarget(NodeT Block) { Target = Block; }

  llvm::SmallPtrSet<NodeT, 4> getReachables() { return Targets; }

  llvm::SmallPtrSet<NodeT, 4> &getAdditional(NodeT Block) {
    return AdditionalNodes[Block];
  }

  // Customize the `insert` method, in order to add the reachables nodes
  // during the DFS.
  std::pair<typename StatusMap::iterator, bool> insert(NodeT Block) {

    // We need to insert the `Source` node, which is the first element on
    // which the `insert` method is called, only once, and later on skip it,
    // otherwise we may loop back from the `Source` and add additional nodes.
    assert(Source != nullptr);
    if (!FirstInvocation and Block == Source) {
      return DFState<NodeT>::insertInMap(Block, false);
    }
    FirstInvocation = false;

    // Check that, if we are trying to insert a block which is the `Targets`
    // set, we add all the nodes on the current visiting stack in the
    // `Targets` set.
    auto BeginIt = llvm::GraphTraits<NodeT>::child_begin(Block);
    auto EndIt = llvm::GraphTraits<NodeT>::child_end(Block);
    size_t BlockSuccessorsN = std::distance(BeginIt, EndIt);
    if ((Targets.contains(Block)) or
        ((Target == nullptr) && (BlockSuccessorsN == 0))) {
      for (auto const &[K, V] : *this) {
        if (V) {
          Targets.insert(K);
        }
      }
    }

    // When we encounter a loop, we add to the additional set of nodes the
    // nodes that are onStack, for later additional post-processing.
    if (DFState<NodeT>::onStack(Block)) {
      llvm::SmallPtrSet<NodeT, 4> &AdditionalSet = AdditionalNodes[Block];
      for (auto const &[K, V] : *this) {
        if (V) {
          AdditionalSet.insert(K);
        }
      }
    }

    // Return the insertion iterator as usual.
    return DFState<NodeT>::insertInMap(Block, true);
  }
};

} // namespace revng::detail

template <class GraphT, class GT = llvm::GraphTraits<GraphT>>
llvm::SmallSet<revng::detail::EdgeDescriptor<typename GT::NodeRef>, 4>
getBackedges(GraphT Block,
             const std::function<bool(const typename GT::NodeRef)> &IsValid) {
  using NodeRef = typename GT::NodeRef;
  using StateType = typename revng::detail::DFSBackedgeState<NodeRef>;
  StateType State(IsValid);

  // Declare manually a custom `df_iterator`
  using bdf_iterator = llvm::df_iterator<GraphT, StateType, true, GT>;
  auto Begin = bdf_iterator::begin(Block, State);
  auto End = bdf_iterator::end(Block, State);

  for (NodeRef Block : llvm::make_range(Begin, End)) {
    for (NodeRef Succ :
         llvm::make_range(GT::child_begin(Block), GT::child_end(Block))) {
      if (State.onStack(Succ)) {
        State.insertBackedge(Block, Succ);
      }
    }
  }

  return State.getBackedges();
}

template <class GraphT, class GT = llvm::GraphTraits<GraphT>>
llvm::SmallSet<revng::detail::EdgeDescriptor<typename GT::NodeRef>, 4>
getBackedges(GraphT Block) {
  return getBackedges(Block, [](const typename GT::NodeRef B) { return true; });
}

template <class GraphT, class GT = llvm::GraphTraits<GraphT>>
llvm::SmallPtrSet<typename GT::NodeRef, 4>
nodesBetweenImpl(GraphT Source, GraphT Target,
                 const llvm::SmallPtrSetImpl<GraphT> *IgnoreList = nullptr) {
  using NodeRef = typename GT::NodeRef;
  using StateType = revng::detail::DFSReachableState<NodeRef>;
  StateType State;

  // If the `IgnoreList` is not empty, populate the ext set with the nodes that
  // it contains.
  if (IgnoreList != nullptr) {
    for (GraphT Element : *IgnoreList) {
      State.insertInMap(Element, false);
    }
  }

  // Assign the `Source` node.
  State.assignSource(Source);

  // Initialize the visited set with the target node, which is the boundary
  // that we don't want to trepass when finding reachable nodes.
  State.assignTarget(Target);
  State.insertTarget(Target);
  State.insertInMap(Target, false);

  using nbdf_iterator = llvm::df_iterator<GraphT, StateType, true, GT>;
  auto Begin = nbdf_iterator::begin(Source, State);
  auto End = nbdf_iterator::end(Source, State);

  for (NodeRef Block : llvm::make_range(Begin, End)) {
    (void)Block;
  }

  auto Targets = State.getReachables();
  // Add in a fixed point fashion the additional nodes.
  llvm::SmallPtrSet<NodeRef, 4> OldTargets;
  do {
    // At each iteration obtain a copy of the old set, so that we are able to
    // exit from the loop as soon no change is made to the `Targets` set.

    OldTargets = Targets;

    // Temporary storage for the nodes to add at each iteration, to avoid
    // invalidation on the `Targets` set.
    llvm::SmallPtrSet<NodeRef, 4> NodesToAdd;

    for (NodeRef Block : Targets) {
      llvm::SmallPtrSet<NodeRef, 4> &AdditionalSet = State.getAdditional(Block);
      NodesToAdd.insert(AdditionalSet.begin(), AdditionalSet.end());
    }

    // Add all the additional nodes found in this step.
    Targets.insert(NodesToAdd.begin(), NodesToAdd.end());
    NodesToAdd.clear();

  } while (Targets != OldTargets);

  return Targets;
}

template <class GraphT>
inline llvm::SmallPtrSet<GraphT, 4>
nodesBetween(GraphT Source, GraphT Destination,
             const llvm::SmallPtrSetImpl<GraphT> *IgnoreList = nullptr) {
  return nodesBetweenImpl<GraphT, llvm::GraphTraits<GraphT>>(
      Source, Destination, IgnoreList);
}

template <class GraphT>
inline llvm::SmallPtrSet<GraphT, 4>
nodesBetweenReverse(GraphT Source, GraphT Destination,
                    const llvm::SmallPtrSetImpl<GraphT> *IgnoreList = nullptr) {
  using namespace llvm;
  return nodesBetweenImpl<GraphT, GraphTraits<Inverse<GraphT>>>(
      Source, Destination, IgnoreList);
}

template <class NodeT>
bool intersect(llvm::SmallPtrSet<NodeT, 4> &First,
               llvm::SmallPtrSet<NodeT, 4> &Second) {

  std::set<NodeT> FirstSet;
  std::set<NodeT> SecondSet;
  FirstSet.insert(First.begin(), First.end());
  SecondSet.insert(Second.begin(), Second.end());

  llvm::SmallVector<NodeT> Intersection;
  std::set_intersection(FirstSet.begin(), FirstSet.end(), SecondSet.begin(),
                        SecondSet.end(), std::back_inserter(Intersection));
  return !Intersection.empty();
}

template <class NodeT>
bool subset(llvm::SmallPtrSet<NodeT, 4> &Contained,
            llvm::SmallPtrSet<NodeT, 4> &Containing) {
  std::set<NodeT> ContainedSet;
  std::set<NodeT> ContainingSet;
  ContainedSet.insert(Contained.begin(), Contained.end());
  ContainingSet.insert(Containing.begin(), Containing.end());
  return std::includes(ContainingSet.begin(), ContainingSet.end(),
                       ContainedSet.begin(), ContainedSet.end());
}

template <class NodeT>
bool equal(llvm::SmallPtrSet<NodeT, 4> &First,
           llvm::SmallPtrSet<NodeT, 4> &Second) {
  std::set<NodeT> FirstSet;
  std::set<NodeT> SecondSet;
  FirstSet.insert(First.begin(), First.end());
  SecondSet.insert(Second.begin(), Second.end());
  return FirstSet == SecondSet;
}

template <class NodeT>
bool simplifyRegionsStep(llvm::SmallVector<llvm::SmallPtrSet<NodeT, 4>> &R) {
  for (auto RegionIt1 = R.begin(); RegionIt1 != R.end(); RegionIt1++) {
    for (auto RegionIt2 = std::next(RegionIt1); RegionIt2 != R.end();
         RegionIt2++) {
      bool Intersects = intersect(*RegionIt1, *RegionIt2);
      bool IsIncluded = subset(*RegionIt1, *RegionIt2);
      bool IsIncludedReverse = subset(*RegionIt2, *RegionIt1);
      bool AreEquivalent = equal(*RegionIt1, *RegionIt2);
      if (Intersects and
          (((!IsIncluded) and (!IsIncludedReverse)) or AreEquivalent)) {
        (*RegionIt1).insert((*RegionIt2).begin(), (*RegionIt2).end());
        R.erase(RegionIt2);
        return true;
      }
    }
  }

  return false;
}

template <class NodeT>
void simplifyRegions(llvm::SmallVector<llvm::SmallPtrSet<NodeT, 4>> &Rs) {
  bool Changes = true;
  while (Changes) {
    Changes = simplifyRegionsStep(Rs);
  }
}

// Reorder the vector containing the regions so that they are in increasing size
// order.
template <class NodeT>
void sortRegions(llvm::SmallVector<llvm::SmallPtrSet<NodeT, 4>> &Rs) {
  std::sort(Rs.begin(), Rs.end(),
            [](llvm::SmallPtrSet<NodeT, 4> &First,
               llvm::SmallPtrSet<NodeT, 4> &Second) {
              return First.size() < Second.size();
            });
}

namespace revng::detail {

template <class NodeT>
class RegionTree;

template <class NodeT>
class RegionNode {
public:
  using NodeRef = std::variant<NodeT, size_t>;

  using succ_container = llvm::SmallVector<RegionNode *>;
  using succ_iterator = typename succ_container::iterator;
  using succ_range = llvm::iterator_range<succ_iterator>;

private:
  using links_container = llvm::SmallVector<NodeRef>;
  using links_iterator = typename links_container::iterator;
  using links_const_iterator = typename links_container::const_iterator;
  using links_range = llvm::iterator_range<links_iterator>;
  using links_const_range = llvm::iterator_range<links_const_iterator>;

  RegionTree<NodeT> &OwningRegionTree;

  // static RegionTree<NodeT> &GlobalRegionTree;

  using getRegionPointerT = RegionNode *(*)(NodeRef &);
  using getConstRegionPointerT = const RegionNode *(*)(const NodeRef &);

  /*
    static RegionNode *getRegionPointer(NodeRef &Successor) {
      assert(std::holds_alternative<size_t>(Successor));
      size_t SuccessorIndex = std::get<size_t>(Successor);
      return &GlobalRegionTree.getRegion(SuccessorIndex);
    }

    static_assert(std::is_same_v<decltype(&getRegionPointer),
    getRegionPointerT>);
  */

private:
  void erase(links_container &V, NodeRef Value) {
    V.erase(std::remove(V.begin(), V.end(), Value), V.end());
  }

  /*
  struct getRegionPointer {
    using result_type = RegionNode *;

    result_type operator()(NodeRef *P) const {
      assert(std::holds_alternative<size_t>(*P));
      size_t SuccessorIndex = std::get<size_t>(*P);
      return &GlobalRegionTree.getRegion(SuccessorIndex);
    }
  };
  */

private:
  links_container Nodes;
  succ_container Children;

public:
  RegionNode(RegionTree<NodeT> &RegionTree) : OwningRegionTree(RegionTree) {}

  links_iterator begin() { return Nodes.begin(); }
  links_const_iterator begin() const { return Nodes.begin(); }
  links_iterator end() { return Nodes.end(); }
  links_const_iterator end() const { return Nodes.end(); }
  links_range regions() { return llvm::make_range(begin(), end()); }
  links_const_range regions() const { return llvm::make_range(begin(), end()); }

  succ_iterator succ_begin() { return Children.begin(); }
  succ_iterator succ_end() { return Children.end(); }

  /*
  using succ_iterator =
      llvm::mapped_iterator<links_iterator, getRegionPointerT>;
  using succ_const_iterator =
      llvm::mapped_iterator<links_const_iterator, getConstRegionPointerT>;
  */

  // static RegionTree<NodeT> *GlobalRegionTree;
  // using succ_iterator = links_iterator;

  /*
  using succ_iterator = typename llvm::SmallVector<RegionNode *>::iterator;
  using succ_range = typename llvm::iterator_range<succ_iterator>;

  links_range getSuccessorsIndex() {
    return llvm::to_vector(llvm::make_filter_range(Nodes, [](NodeRef &Node)
  { return std::holds_alternative<size_t>(Node);
    }));
  }

  links_const_range getSuccessorsIndex() const {
    return llvm::to_vector(llvm::make_filter_range(Nodes, [](NodeRef &Node)
  { return std::holds_alternative<size_t>(Node);
    }));
  }
  */

  /*
    succ_iterator succ_begin() {
      return llvm::map_iterator(getSuccessorsIndex().begin(),
    getRegionPointer());
    }

    succ_iterator succ_end() {
      return llvm::map_iterator(getSuccessorsIndex().end(),
    getRegionPointer());
    }
  */

  /*
  links_range getSuccessors() {
    return llvm::to_vector([this](size_t RegionIndex) {
      return OwningRegionTree.getRegion(RegionIndex);
    });
  }
  */

  /*
  succ_range getSuccessors() {
    llvm::to_vector(llvm::map_range(getSuccessorsIndex(), [&](size_t &Index)
  { return OwningRegionTree.getRegion(Index);
    }));

    llvm::SmallVector<RegionNode *> SuccessorsRegions;
    for (auto SuccessorIndex : getSuccessorsIndex()) {
      assert(std::holds_alternative<size_t>(SuccessorIndex));
      size_t SuccessorIndexSizeT = std::get<size_t>(SuccessorIndex);
      SuccessorsRegions.push_back(
          &(OwningRegionTree.getRegion(SuccessorIndexSizeT)));
    }

    return llvm::make_range(SuccessorsRegions.begin(),
  SuccessorsRegions.end());
  }

  succ_iterator succ_begin() { return getSuccessors().begin(); }
  succ_iterator succ_end() { return getSuccessors().end(); }
  */

  /*
  succ_iterator succ_begin() {
    return llvm::map_iterator(getSuccessorsIndex().begin(),
  getRegionPointer);
  }

  succ_iterator succ_end() {
    return llvm::map_iterator(getSuccessorsIndex().end(), getRegionPointer);
  }
  */

  /*
  succ_iterator succ_begin() {
    return llvm::map_iterator(
        getSuccessorsIndex().begin(), [&](NodeRef &Successor) {
          assert(std::holds_alternative<size_t>(Successor));
          size_t SuccessorIndex = std::get<size_t>(Successor);
          return &OwningRegionTree.getRegion(SuccessorIndex);
        });
  }

  succ_const_iterator succ_begin() const {
    return llvm::map_iterator(
        getSuccessorsIndex().begin(), [&](const NodeRef &Successor) {
          assert(std::holds_alternative<size_t>(Successor));
          size_t SuccessorIndex = std::get<size_t>(Successor);
          return &OwningRegionTree.getRegion(SuccessorIndex);
        });
  }

  succ_iterator succ_end() {
    return llvm::map_iterator(
        getSuccessorsIndex().end(), [&](NodeRef &Successor) {
          assert(std::holds_alternative<size_t>(Successor));
          size_t SuccessorIndex = std::get<size_t>(Successor);
          return &OwningRegionTree.getRegion(SuccessorIndex);
        });
  }

  succ_const_iterator succ_end() const {
    return llvm::map_iterator(
        getSuccessorsIndex().end(), [&](const NodeRef &Successor) {
          assert(std::holds_alternative<size_t>(Successor));
          size_t SuccessorIndex = std::get<size_t>(Successor);
          return &OwningRegionTree.getRegion(SuccessorIndex);
        });
  }
  */

  // Insert helpers.
  void insertChildRegion(NodeRef Element) {
    assert(std::holds_alternative<size_t>(Element));
    size_t RegionIndex = std::get<size_t>(Element);
    RegionNode *ChildRegion = &OwningRegionTree.getRegion(RegionIndex);
    Children.push_back(ChildRegion);
  }

  void insertElement(NodeRef Element) {
    Nodes.push_back(Element);
    if (std::holds_alternative<size_t>(Element)) {
      insertChildRegion(Element);
    }
  }
  void insertElementEntry(NodeRef Element) {
    Nodes.insert(begin(), Element);
    if (std::holds_alternative<size_t>(Element)) {
      insertChildRegion(Element);
    }
  }

  // If we are removing the first element (hardcoded entry), we signal it with
  // the return code.
  bool eraseElement(NodeRef Element) {
    bool IsEntry = Nodes.front() == Element;
    erase(Nodes, Element);
    return IsEntry;
  }
};

// TODO: double check how to implement the variant with the fact that we want to
// accept a template argument for the node type, but have an index at the same
// time.
template <class NodeT>
class RegionTree {
  using RegionVector = RegionNode<NodeT>;

public:
  using links_container = llvm::SmallVector<RegionVector>;
  using links_iterator = typename links_container::iterator;
  using links_const_iterator = typename links_container::const_iterator;
  using links_reverse_iterator = typename links_container::reverse_iterator;
  using links_const_reverse_iterator =
      typename links_container::const_reverse_iterator;
  using links_range = llvm::iterator_range<links_iterator>;
  using links_reverse_range = llvm::iterator_range<links_reverse_iterator>;
  using links_const_range = llvm::iterator_range<links_const_iterator>;
  using links_const_reverse_range =
      llvm::iterator_range<links_const_reverse_iterator>;

private:
  links_container Regions;

public:
  RegionTree() = default;

  void insertRegion(RegionVector &&Region) {
    Regions.emplace_back(std::move(Region));
  }

  RegionVector &front() { return Regions.front(); }

  links_iterator begin() { return Regions.begin(); }
  links_const_iterator begin() const { return Regions.begin(); }
  links_iterator end() { return Regions.end(); }
  links_const_iterator end() const { return Regions.end(); }
  links_range regions() { return llvm::make_range(begin(), end()); }
  links_const_range regions() const { return llvm::make_range(begin(), end()); }

  links_reverse_iterator rbegin() { return Regions.rbegin(); }
  links_const_reverse_iterator rbegin() const { return Regions.rbegin(); }
  links_reverse_iterator rend() { return Regions.rend(); }
  links_const_reverse_iterator rend() const { return Regions.rend(); }
  links_reverse_range reverse_regions() {
    return llvm::make_range(rbegin(), rend());
  }
  links_const_reverse_range reverse_regions() const {
    return llvm::make_range(rbegin(), rend());
  }

  RegionVector &getRegion(size_t Index) { return Regions[Index]; }
};

using ParentMap = llvm::DenseMap<std::ptrdiff_t, std::ptrdiff_t>;

// TODO: this data structure will be responsible of handling the child/parent
//       relationship of identified regions. We now implemented this with
//       keeping indexes on the underlying vector around, but in future we may
//       want to move the ownership inside and expose `GraphTraits`.
template <class NodeT>
class ParentTree {
  using ParentMap = llvm::DenseMap<std::ptrdiff_t, std::ptrdiff_t>;
  using RegionSet = llvm::SmallPtrSet<NodeT, 4>;

  using links_container = llvm::SmallVector<RegionSet>;
  using links_iterator = typename links_container::iterator;
  using links_const_iterator = typename links_container::const_iterator;
  using links_range = llvm::iterator_range<links_iterator>;
  using links_const_range = llvm::iterator_range<links_const_iterator>;

private:
  ParentMap Map;
  links_container Rs;
  llvm::DenseMap<std::ptrdiff_t, bool> IsRootRegionMap;
  llvm::DenseMap<std::ptrdiff_t, NodeT> EntryMap;

  // TODO: this field is not used, implement the check that any query to the
  // data structure find it in a ready state.
  bool ReadyState = false;

private:
  RegionSet &getRegionFromIndex(std::ptrdiff_t Index) { return Rs[Index]; }

  void computeParents() {
    for (auto RegionIt1 = Rs.begin(); RegionIt1 != Rs.end(); RegionIt1++) {
      for (auto RegionIt2 = std::next(RegionIt1); RegionIt2 != Rs.end();
           RegionIt2++) {
        if (subset(*RegionIt1, *RegionIt2)) {
          Map[getRegionIndex(*RegionIt1)] = getRegionIndex(*RegionIt2);
          break;
        }
      }
    }
  }

public:
  ParentTree() = default;

  void clear() {
    Map.clear();
    Rs.clear();
    IsRootRegionMap.clear();
    EntryMap.clear();
    ReadyState = false;
  }

  links_container &getRegions() { return Rs; }

  std::ptrdiff_t getRegionIndex(RegionSet &Region) {
    for (auto RegionIt = Rs.begin(); RegionIt != Rs.end(); RegionIt++) {
      if (*RegionIt == Region) {
        return std::distance(Rs.begin(), RegionIt);
      }
    }

    // TODO: We may want to soft fail in this situation, if we allow to query
    //       the data structure with no assurance that the intended region is
    //       present.
    assert(false);
  }

  // TODO: we need this method because we cannot have `std::optional` with
  //       references.
  bool hasParent(RegionSet &Child) {
    std::ptrdiff_t ChildIndex = getRegionIndex(Child);
    auto MapIt = Map.find(ChildIndex);
    return MapIt != Map.end();
  }

  RegionSet &getParent(RegionSet &Child) {
    std::ptrdiff_t ChildIndex = getRegionIndex(Child);
    auto MapIt = Map.find(ChildIndex);
    assert(MapIt != Map.end());
    std::ptrdiff_t ParentIndex = MapIt->second;
    RegionSet &Parent = getRegionFromIndex(ParentIndex);
    return Parent;
  }

  void insertRegion(RegionSet &Region) { Rs.emplace_back(std::move(Region)); }

  void order() {
    computeParents();
    computePartialOrder();
    computeParents();
  }

  links_iterator begin() { return Rs.begin(); }
  links_const_iterator begin() const { return Rs.begin(); }
  links_iterator end() { return Rs.end(); }
  links_const_iterator end() const { return Rs.end(); }
  links_range regions() { return llvm::make_range(begin(), end()); }
  links_const_range regions() const { return llvm::make_range(begin(), end()); }

  void computePartialOrder() {
    links_container OrderedRegions;
    llvm::SmallPtrSet<size_t, 4> Processed;

    while (Rs.size() != Processed.size()) {
      for (auto RegionIt1 = begin(); RegionIt1 != end(); RegionIt1++) {
        if (Processed.count(getRegionIndex(*RegionIt1)) == 0) {
          bool FoundParent = false;
          for (auto RegionIt2 = std::next(RegionIt1); RegionIt2 != Rs.end();
               RegionIt2++) {
            if (Processed.count(getRegionIndex(*RegionIt2)) == 0) {
              if (getParent(*RegionIt1) == *RegionIt2) {
                FoundParent = true;
                break;
              }
            }
          }

          if (FoundParent == false) {
            OrderedRegions.push_back(*RegionIt1);
            Processed.insert(getRegionIndex(*RegionIt1));
            break;
          }
        }
      }
    }

    // Swap the region vector with the ordered one.
    std::reverse(OrderedRegions.begin(), OrderedRegions.end());
    Rs.swap(OrderedRegions);
  }

  void setRegionRoot(RegionSet &Region, bool Value) {
    std::ptrdiff_t RegionIndex = getRegionIndex(Region);
    IsRootRegionMap[RegionIndex] = Value;
  }

  bool isRegionRoot(RegionSet &Region) {
    std::ptrdiff_t RegionIndex = getRegionIndex(Region);
    auto MapIt = IsRootRegionMap.find(RegionIndex);
    assert(MapIt != IsRootRegionMap.end());
    return MapIt->second;
  }

  void setRegionEntry(RegionSet &Region, NodeT Entry) {
    std::ptrdiff_t RegionIndex = getRegionIndex(Region);
    EntryMap[RegionIndex] = Entry;
  }

  NodeT getRegionEntry(RegionSet &Region) {
    std::ptrdiff_t RegionIndex = getRegionIndex(Region);
    auto MapIt = EntryMap.find(RegionIndex);
    assert(MapIt != EntryMap.end());
    return MapIt->second;
  }
};

} // namespace revng::detail

namespace llvm {
template <>
struct GraphTraits<revng::detail::RegionNode<mlir::Block *> *> {
  using ChildIteratorType =
      revng::detail::RegionNode<mlir::Block *>::succ_iterator;
  using Node = revng::detail::RegionNode<mlir::Block *>;
  using NodeRef = Node *;

  static NodeRef getEntryNode(NodeRef BB) { return BB; }

  static ChildIteratorType child_begin(NodeRef Node) {
    return Node->succ_begin();
  }
  static ChildIteratorType child_end(NodeRef Node) { return Node->succ_end(); }
};

template <>
struct GraphTraits<revng::detail::RegionTree<mlir::Block *> *>
    : public GraphTraits<revng::detail::RegionNode<mlir::Block *> *> {
  using GraphType = revng::detail::RegionTree<mlir::Block *> *;
  using NodeRef = revng::detail::RegionNode<mlir::Block *> *;

  static NodeRef getEntryNode(GraphType Rt) { return &Rt->front(); }

  using nodes_iterator =
      revng::detail::RegionTree<mlir::Block *>::links_iterator;
  static nodes_iterator nodes_begin(GraphType Rt) {
    return nodes_iterator(Rt->begin());
  }
  static nodes_iterator nodes_end(GraphType Rt) {
    return nodes_iterator(Rt->end());
  }
};

} // namespace llvm

template <class GraphT, class GT = llvm::GraphTraits<GraphT>,
          typename NodeRef = typename GT::NodeRef>
llvm::DenseMap<NodeRef, size_t> computeDistanceFromEntry(GraphT Source) {
  llvm::DenseMap<NodeRef, size_t> ShortestPathFromEntry;

  using SetType = llvm::bf_iterator_default_set<NodeRef>;
  using bf_iterator = llvm::bf_iterator<GraphT, SetType, GT>;
  auto BFSIt = bf_iterator::begin(Source);
  auto BFSEnd = bf_iterator::end(Source);

  for (; BFSIt != BFSEnd; ++BFSIt) {
    NodeRef Block = *BFSIt;
    size_t Depth = BFSIt.getLevel();

    // Obtain the insertion iterator for the `Depth` block element.
    auto ShortestIt = ShortestPathFromEntry.insert({Block, Depth});

    // If we already had in the map an entry for the current block, we need to
    // assert that the previously found value for the `Depth` is less or equal
    // of the `Depth` we are inserting.
    if (ShortestIt.second == false) {
      assert(ShortestIt.first->second <= Depth);
    }
  }

  return ShortestPathFromEntry;
}

template <class NodeT>
bool setContains(llvm::SmallPtrSetImpl<NodeT> &Set, NodeT &Element) {
  return Set.contains(Element);
}

template <class GraphT, class GT = llvm::GraphTraits<GraphT>>
auto child_range(GraphT Block) {
  return llvm::make_range(GT::child_begin(Block), GT::child_end(Block));
}

template <class GraphT>
auto successor_range(GraphT Block) {
  return child_range(Block);
}

template <class GraphT>
auto predecessor_range(GraphT Block) {
  return child_range<GraphT, llvm::GraphTraits<llvm::Inverse<GraphT>>>(Block);
}

template <class NodeRef>
llvm::DenseMap<NodeRef, size_t>
getEntryCandidates(llvm::SmallPtrSetImpl<NodeRef> &Region) {

  // `DenseMap` that will contain all the candidate entries of a region, with
  // the associated incoming edges degree.
  llvm::DenseMap<NodeRef, size_t> Result;

  // We can iterate over all the predecessors of a block, if we find a pred not
  // in the current set, we increment the counter of the entry edges.
  for (NodeRef Block : Region) {
    for (NodeRef Predecessor : predecessor_range(Block)) {
      if (not setContains(Region, Predecessor)) {
        Result[Block]++;
      }
    }
  }

  return Result;
}

template <class NodeT>
size_t mapAt(llvm::DenseMap<NodeT, size_t> &Map, NodeT Element) {
  auto MapIt = Map.find(Element);
  assert(MapIt != Map.end());
  return MapIt->second;
}

template <class NodeT>
NodeT electEntry(llvm::DenseMap<NodeT, size_t> &EntryCandidates,
                 llvm::DenseMap<NodeT, size_t> &ShortestPathFromEntry,
                 llvm::SmallVectorImpl<NodeT> &RPOT) {
  // Elect the Entry as the the candidate entry with the largest number of
  // incoming edges from outside the region.
  // If there's a tie, i.e. there are 2 or more candidate entries with the
  // same number of incoming edges from an outer region, we select the entry
  // with the minimal shortest path from entry.
  // It it's still a tie, i.e. there are 2 or more candidate entries with the
  // same number of incoming edges from an outer region and the same minimal
  // shortest path from entry, then we disambiguate by picking the entry that
  // comes first in RPOT.
  NodeT Entry = Entry = EntryCandidates.begin()->first;
  {
    size_t MaxNEntries = EntryCandidates.begin()->second;
    auto ShortestPathIt = ShortestPathFromEntry.find(Entry);
    assert(ShortestPathIt != ShortestPathFromEntry.end());
    size_t ShortestPath = mapAt(ShortestPathFromEntry, Entry);
    auto EntriesEnd = EntryCandidates.end();
    for (NodeT Block : RPOT) {
      auto EntriesIt = EntryCandidates.find(Block);
      if (EntriesIt != EntriesEnd) {
        const auto &[EntryCandidate, NumIncoming] = *EntriesIt;
        if (NumIncoming > MaxNEntries) {
          Entry = EntryCandidate;
          ShortestPath = mapAt(ShortestPathFromEntry, EntryCandidate);
        } else if (NumIncoming == MaxNEntries) {
          size_t SP = mapAt(ShortestPathFromEntry, EntryCandidate);
          if (SP < ShortestPath) {
            Entry = EntryCandidate;
            ShortestPath = SP;
          }
        }
      }
    }
  }
  assert(Entry != nullptr);
  return Entry;
}

template <class NodeRef>
llvm::SmallVector<std::pair<NodeRef, NodeRef>>
getOutlinedEntries(llvm::DenseMap<NodeRef, size_t> &EntryCandidates,
                   llvm::SmallPtrSetImpl<NodeRef> &Region, NodeRef Entry) {
  llvm::SmallVector<std::pair<NodeRef, NodeRef>> LateEntryPairs;
  for (const auto &[Other, NumIncoming] : EntryCandidates) {
    if (Other != Entry) {
      llvm::SmallVector<NodeRef> OutsidePredecessor;
      for (NodeRef Predecessor : predecessor_range(Other)) {
        if (not setContains(Region, Predecessor)) {
          OutsidePredecessor.push_back(Predecessor);
          LateEntryPairs.push_back({Predecessor, Other});
        }
      }
      assert(OutsidePredecessor.size() == NumIncoming);
    }
  }

  return LateEntryPairs;
}

template <class NodeRef>
llvm::SmallVector<std::pair<NodeRef, NodeRef>>
getExitNodePairs(llvm::SmallPtrSetImpl<NodeRef> &Region) {

  // Vector that contains the pairs of exit/successor node pairs.
  llvm::SmallVector<std::pair<NodeRef, NodeRef>> ExitSuccessorPairs;

  // We iterate over all the successors of a block, if we find a successor not
  // in the current set, we add the pairs of node to the set.
  for (NodeRef Block : Region) {
    for (NodeRef Successor : successor_range(Block)) {
      if (not setContains(Region, Successor)) {
        ExitSuccessorPairs.push_back({Block, Successor});
      }
    }
  }

  return ExitSuccessorPairs;
}

template <class NodeRef>
llvm::SmallVector<std::pair<NodeRef, NodeRef>>
getPredecessorNodePairs(NodeRef Node) {
  llvm::SmallVector<std::pair<NodeRef, NodeRef>> PredecessorNodePairs;
  for (NodeRef Predecessor : predecessor_range(Node)) {
    PredecessorNodePairs.push_back({Predecessor, Node});
  }

  return PredecessorNodePairs;
}

template <class NodeRef>
llvm::SmallVector<std::pair<NodeRef, NodeRef>>
getLoopPredecessorNodePairs(NodeRef Node,
                            llvm::SmallPtrSetImpl<NodeRef> &Region) {
  llvm::SmallVector<std::pair<NodeRef, NodeRef>> LoopPredecessorNodePairs;
  for (NodeRef Predecessor : predecessor_range(Node)) {
    if (not setContains(Region, Predecessor)) {
      LoopPredecessorNodePairs.push_back({Predecessor, Node});
    }
  }

  return LoopPredecessorNodePairs;
}

template <class NodeRef>
llvm::SmallVector<std::pair<NodeRef, NodeRef>>
getContinueNodePairs(NodeRef Entry, llvm::SmallPtrSetImpl<NodeRef> &Region) {
  llvm::SmallVector<std::pair<NodeRef, NodeRef>> ContinueNodePairs;
  for (NodeRef Predecessor : predecessor_range(Entry)) {
    if (setContains(Region, Predecessor)) {
      ContinueNodePairs.push_back({Predecessor, Entry});
    }
  }

  return ContinueNodePairs;
}
