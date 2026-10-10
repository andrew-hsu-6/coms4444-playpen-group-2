import math
import time
from functools import lru_cache
from itertools import product

from players.player0 import Player
from shapely.geometry import Polygon
from shapely.prepared import prep
from src.constants import BASELINE_SCORE, MAX_FACE_LENGTH, TOL
from src.enclosure import Construction, score_construction, validate_construction
from src.pieces import Connector, ConnectorType, Piece, PieceType


class Player2(Player):
    """Search rectangles, octagons, L-shapes, and 90/135-degree hexagons."""

    # Leave ample headroom under the simulator's default 300 CPU seconds.
    # Lower this when testing with a smaller --cpu-limit.
    SEARCH_CPU_SECONDS = 60.0
    ALLOCATION_NODE_LIMIT = 20000

    def build_enclosure(self) -> Construction | None:
        deadline = time.process_time() + self.SEARCH_CPU_SECONDS
        best = None
        best_score = 0.0  # Returning None is preferable to a negative score.
        straight = self.inventory.connectors.get(ConnectorType.STRAIGHT, 0)

        types = []
        stock = []
        for kind, pool in (
            (PieceType.WALL, self.inventory.walls),
            (PieceType.GATE, self.inventory.gates),
        ):
            for length, count in sorted(pool.items()):
                if count > 0 and length <= MAX_FACE_LENGTH:
                    types.append(Piece(kind, length))
                    stock.append(count)
        if not any(piece.is_gate for piece in types):
            self.note = "No usable gate of length at most 30."
            return None

        # Enumerate unordered face assemblies. Each face contains at most six
        # pieces; preserve different resource allocations for the same length.
        faces = {}
        counts = [0] * len(types)

        def enumerate_faces(first, length, number, has_gate):
            if time.process_time() >= deadline:
                return
            if number:
                faces.setdefault(length, []).append(
                    (tuple(counts), number - 1, has_gate)
                )
            if number >= min(6, straight + 1):
                return
            for i in range(first, len(types)):
                new_length = length + types[i].length
                if new_length > MAX_FACE_LENGTH or counts[i] >= stock[i]:
                    continue
                counts[i] += 1
                enumerate_faces(i, new_length, number + 1, has_gate or types[i].is_gate)
                counts[i] -= 1

        enumerate_faces(0, 0, 0, False)
        for options in faces.values():
            options.sort(key=lambda option: (option[1], not option[2]))

        def allocate(lengths, required_gate_faces):
            """Bounded backtracking with shared inventory and connector counts."""
            nodes = 0

            @lru_cache(maxsize=20000)
            def search(side, remaining, available_straight, gate_faces):
                nonlocal nodes
                nodes += 1
                if (
                    nodes > self.ALLOCATION_NODE_LIMIT
                    or time.process_time() >= deadline
                ):
                    return None
                if side == len(lengths):
                    return () if gate_faces >= required_gate_faces else None
                if gate_faces + len(lengths) - side < required_gate_faces:
                    return None
                for usage, joins, has_gate in faces[lengths[side]]:
                    if joins > available_straight:
                        continue
                    if any(used > left for used, left in zip(usage, remaining)):
                        continue
                    rest = search(
                        side + 1,
                        tuple(left - used for used, left in zip(usage, remaining)),
                        available_straight - joins,
                        min(required_gate_faces, gate_faces + int(has_gate)),
                    )
                    if rest is not None:
                        return (usage,) + rest
                return None

            return search(0, tuple(stock), straight, 0)

        room = self.room.polygon
        buffered_room = prep(room.buffer(TOL))
        corners = list(room.exterior.coords)[:-1]
        headings = set(range(0, 180, 15))
        for p, q in zip(corners, corners[1:] + corners[:1]):
            angle = math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))
            headings.add(round(angle % 180, 9))
            headings.add(round((angle + 90) % 180, 9))
        centers = [(room.centroid.x, room.centroid.y)]
        inside = room.representative_point()
        centers.append((inside.x, inside.y))
        minx, miny, maxx, maxy = room.bounds
        for i in range(1, 10):
            for j in range(1, 10):
                centers.append(
                    (minx + (maxx - minx) * i / 10, miny + (maxy - miny) * j / 10)
                )

        def placement(vertices):
            local = Polygon(vertices)
            center = local.centroid
            # Non-symmetric templates need the full circle. Include alignment
            # of EVERY template edge with EVERY room edge, not just edge zero.
            angles = set(range(0, 360, 15))
            for heading in headings:
                for offset in range(0, 360, 45):
                    angles.add(round((heading + offset) % 360, 9))
            for heading in sorted(angles):
                if time.process_time() >= deadline:
                    return None
                theta = math.radians(heading)
                co, si = math.cos(theta), math.sin(theta)
                offsets = [(x * co - y * si, x * si + y * co) for x, y in vertices]
                cx = center.x * co - center.y * si
                cy = center.x * si + center.y * co
                starts = [
                    (px - ox, py - oy) for px, py in corners for ox, oy in offsets
                ]
                starts.extend((x - cx, y - cy) for x, y in centers)
                for x, y in starts:
                    if time.process_time() >= deadline:
                        return None
                    polygon = Polygon([(x + dx, y + dy) for dx, dy in offsets])
                    if buffered_room.contains(polygon):
                        return (x, y), heading
            return None

        # Face lengths are always sums of INTEGER physical piece lengths.
        # Enumerate integer parameters; never round a diagonal's length from
        # coordinates. Opposite equal faces make diagonal templates close.
        lengths = sorted(faces)
        right = self.inventory.connectors.get(ConnectorType.RIGHT, 0)
        diagonal = self.inventory.connectors.get(ConnectorType.DIAGONAL, 0)
        room_diagonal = math.hypot(maxx - minx, maxy - miny)
        candidates = []
        generation_deadline = time.process_time() + max(
            0.01, (deadline - time.process_time()) * 0.2
        )

        def templates():
            generators = []
            if right >= 4:
                generators.append(
                    (
                        ("rectangle", (a, b, a, b))
                        for a, b in product(lengths, repeat=2)
                        if a <= b
                    )
                )
            if right >= 2 and diagonal >= 4:
                generators.append(
                    (
                        ("hexagon", (a, b, c, a, b, c))
                        for a, b, c in product(lengths, repeat=3)
                    )
                )
            if diagonal >= 8:
                generators.append(
                    (
                        ("octagon", (a, b, c, d, a, b, c, d))
                        for a, b, c, d in product(lengths, repeat=4)
                        if a <= c
                    )
                )
            if right >= 6:
                generators.append(
                    (
                        ("L-shape", (a + b, c, b, d, a, c + d))
                        for a, b, c, d in product(lengths, repeat=4)
                        if a + b in faces and c + d in faces
                    )
                )
            # Round-robin enumeration gives each family search time.
            while generators and time.process_time() < generation_deadline:
                for generator in generators[:]:
                    try:
                        yield next(generator)
                    except StopIteration:
                        generators.remove(generator)

        available_perimeter = sum(p.length * n for p, n in zip(types, stock))
        for family, sides in templates():
            if sum(sides) > available_perimeter:
                continue
            if family == "rectangle":
                area = sides[0] * sides[1]
            elif family == "hexagon":
                a, b, c = sides[:3]
                area = a * c + (a + c) * b / math.sqrt(2)
            elif family == "octagon":
                a, b, c, d = sides[:4]
                area = a * c + (a + c) * (b + d) / math.sqrt(2) + b * d
            else:
                width, c, b, d, a, _ = sides
                area = width * c + a * d
            if area > room.area + TOL * room.length:
                continue
            base = BASELINE_SCORE + self.weights.A * area + self.weights.C * sum(sides)
            if base + self.weights.G > 0:
                candidates.append((base + self.weights.G, base, family, sides))
        candidates.sort(reverse=True)

        best_family = ""
        for upper_score, base, family, sides in candidates:
            if time.process_time() >= deadline or upper_score <= best_score:
                break
            turns = self._template_turns(family)
            vertices = self._integer_vertices(sides, turns)
            if vertices is None:
                continue
            shape = Polygon(vertices)
            if not shape.is_valid or shape.area <= 0:
                continue
            # A shape's diameter cannot exceed the room bounding-box diagonal.
            if any(
                math.hypot(x - xx, y - yy) > room_diagonal + TOL
                for x, y in vertices
                for xx, yy in vertices
            ):
                continue
            # Geometry is unchanged by the face assembly or choice of gates.
            assemblies = None
            if self.weights.G > 0:
                assemblies = allocate(sides, 2)
            if assemblies is None and base > best_score:
                assemblies = allocate(sides, 1)
            if assemblies is None:
                continue
            bonus = (
                sum(
                    any(types[i].is_gate and n for i, n in enumerate(face))
                    for face in assemblies
                )
                >= 2
            )
            if base + (self.weights.G if bonus else 0) <= best_score:
                continue
            found = placement(vertices)
            if found is None:
                continue
            pieces, connectors = [], []
            for usage, turn in zip(assemblies, turns):
                face = [piece for piece, n in zip(types, usage) for _ in range(n)]
                pieces.extend(face)
                # The connector belongs to the START of its outgoing piece.
                connectors.append(
                    Connector(
                        ConnectorType.RIGHT
                        if abs(turn) == 90
                        else ConnectorType.DIAGONAL,
                        reflex=turn < 0,
                    )
                )
                connectors.extend(Connector(ConnectorType.STRAIGHT) for _ in face[1:])
            start, heading = found
            construction = Construction(start, heading, pieces, connectors)
            result = validate_construction(construction, self.room, self.inventory)
            if result.valid:
                score = score_construction(result, self.weights)
                if score > best_score:
                    best, best_score = construction, score
                    best_family = family
        self.note = (
            f"Composite {best_family}; score {best_score:.2f}."
            if best
            else "No positive-score template found within search budget."
        )
        return best

    @staticmethod
    def _template_turns(family):
        # Turns are exterior angles, indexed at the START of each face.
        return {
            "rectangle": (90, 90, 90, 90),
            "hexagon": (90, 45, 45, 90, 45, 45),
            "octagon": (45,) * 8,
            "L-shape": (90, 90, 90, -90, 90, 90),
        }[family]

    @staticmethod
    def _integer_vertices(lengths, turns):
        """Check closure exactly in Z + Z/sqrt(2), then emit float vertices.

        Integer lengths at multiples of 45 degrees give coordinates
        (axis_integer + diagonal_integer/sqrt(2)). Both coefficients must
        separately return to zero; arbitrary rounding cannot create closure.
        """
        if (
            len(lengths) != len(turns)
            or sum(turns) != 360
            or any(
                not isinstance(n, int) or not 5 <= n <= MAX_FACE_LENGTH for n in lengths
            )
        ):
            return None
        axis_x = axis_y = diagonal_x = diagonal_y = 0
        direction = 0
        vertices = []
        directions = (
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
            (0, -1),
            (1, -1),
        )
        for i, length in enumerate(lengths):
            if i:
                direction = (direction + turns[i] // 45) % 8
            vertices.append(
                (axis_x + diagonal_x / math.sqrt(2), axis_y + diagonal_y / math.sqrt(2))
            )
            dx, dy = directions[direction]
            if direction % 2:
                diagonal_x += length * dx
                diagonal_y += length * dy
            else:
                axis_x += length * dx
                axis_y += length * dy
        if (axis_x, axis_y, diagonal_x, diagonal_y) != (0, 0, 0, 0):
            return None
        return vertices
