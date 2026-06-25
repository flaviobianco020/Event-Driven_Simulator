from __future__ import annotations
from collections import deque
from enum import Enum
from .node import Node
from .link import Link
from .queue_manager import QueueManager


class TopologyType(Enum):
    SINGLE_BOTTLENECK = "single_bottleneck"
    MULTI_HOP = "multi_hop"
    MESH = "mesh"


class NetworkTopology:
    def __init__(self, topo_type: TopologyType) -> None:
        self.type = topo_type
        self.nodes: list[Node] = []
        self.links: list[Link] = []

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)

    def add_link(self, link: Link) -> None:
        self.links.append(link)

    def get_node(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(f"Node '{node_id}' not found")

    def get_link(self, source_id: str, target_id: str) -> Link:
        for lnk in self.links:
            if lnk.source.id == source_id and lnk.target.id == target_id:
                return lnk
        raise KeyError(f"Link '{source_id}→{target_id}' not found")

    def get_path(self, src_id: str, dst_id: str) -> list[Link]:
        if src_id == dst_id:
            return []
        adj: dict[str, list[Link]] = {}
        for lnk in self.links:
            adj.setdefault(lnk.source.id, []).append(lnk)
        queue: deque[tuple[str, list[Link]]] = deque([(src_id, [])])
        visited = {src_id}
        while queue:
            current, path = queue.popleft()
            for lnk in adj.get(current, []):
                if lnk.target.id not in visited:
                    new_path = path + [lnk]
                    if lnk.target.id == dst_id:
                        return new_path
                    visited.add(lnk.target.id)
                    queue.append((lnk.target.id, new_path))
        return []

    @classmethod
    def single_bottleneck(
        cls,
        n_sources: int = 2,
        bottleneck_capacity: float = 10.0,
        bottleneck_delay: float = 0.005,
        queue_size: int = 20,
    ) -> "NetworkTopology":
        topo = cls(TopologyType.SINGLE_BOTTLENECK)
        dst = Node(id="dst")
        router = Node(
            id="router",
            queues=[QueueManager(max_size=queue_size, service_rate=bottleneck_capacity)],
        )
        topo.add_node(dst)
        topo.add_node(router)
        topo.add_link(Link(source=router, target=dst, capacity=bottleneck_capacity, delay=bottleneck_delay))
        for i in range(n_sources):
            src = Node(id=f"src{i}")
            topo.add_node(src)
            topo.add_link(Link(source=src, target=router, capacity=1000.0, delay=0.001))
        return topo

    @classmethod
    def mesh(
        cls,
        rows: int = 2,
        cols: int = 3,
        capacity: float = 10.0,
        delay: float = 0.005,
        queue_size: int = 20,
    ) -> "NetworkTopology":
        topo = cls(TopologyType.MESH)
        nodes: list[list[Node]] = []
        for r in range(rows):
            row = []
            for c in range(cols):
                n = Node(
                    id=f"n{r}{c}",
                    queues=[QueueManager(max_size=queue_size, service_rate=capacity)],
                )
                topo.add_node(n)
                row.append(n)
            nodes.append(row)
        for r in range(rows):
            for c in range(cols):
                if c + 1 < cols:
                    topo.add_link(Link(nodes[r][c], nodes[r][c + 1], capacity, delay))
                    topo.add_link(Link(nodes[r][c + 1], nodes[r][c], capacity, delay))
                if r + 1 < rows:
                    topo.add_link(Link(nodes[r][c], nodes[r + 1][c], capacity, delay))
                    topo.add_link(Link(nodes[r + 1][c], nodes[r][c], capacity, delay))
        return topo

    @classmethod
    def multi_hop(
        cls,
        n_hops: int = 3,
        capacity: float = 10.0,
        delay: float = 0.005,
        queue_size: int = 20,
    ) -> "NetworkTopology":
        topo = cls(TopologyType.MULTI_HOP)
        nodes = [
            Node(id=f"n{i}", queues=[QueueManager(max_size=queue_size, service_rate=capacity)])
            for i in range(n_hops + 1)
        ]
        for n in nodes:
            topo.add_node(n)
        for i in range(n_hops):
            topo.add_link(Link(source=nodes[i], target=nodes[i + 1], capacity=capacity, delay=delay))
        return topo
