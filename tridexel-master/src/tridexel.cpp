#include "tridexel.h"

#include "IO.h"

#include <numeric>
#include <optional>
#include <string>

namespace {
	// 三维向量
	using glm::vec3;
	// 无符号三维向量
	using glm::uvec3;


	// 表示一个平面。它有两个构造函数，分别用于根据法线和距离、以及法线和一个点来初始化平面。
	struct Plane {
		Plane(vec3 n, float d) : n(n), d(d) {}
		Plane(vec3 n, vec3 p) : n(n), d(glm::dot(n, p)) {}

		// vec3 n: 表示平面的法线向量。
		// float d: 表示平面距离原点的距离。
		vec3 n;
		float d;
	};

	// 定义了一个名为 Node 的结构体，表示 TriDexel 算法中 Dexel 的节点（Node）数据结构。每个节点存储了一个深度（depth）值和一个三维向量（vec3）作为法线（normal）。
	struct Node {
		float depth;
		vec3 normal;
	};

	// 定义了一个名为 Segment 的结构体，表示 TriDexel 算法中的线段（Segment）数据结构。每个线段由起始节点（start）和终止节点（end）组成，用于存储 Dexel 中的线段信息。
	struct Segment {
		Node start;
		Node end;
	};
	// 定义了一个名为 Dexel 的结构体，表示 TriDexel 算法中的 Dexel 数据结构。Dexel 是一种用于存储体素的信息的数据结构，包含了一些用于处理 Dexel 数据的方法。
	struct Dexel {
		void regularize() {
			// TODO: improve this
			// 这是一个成员函数，用于对 Dexel 进行规范化（regularization）。规范化的过程中，如果 nodes 向量中的元素数量为奇数，就会清空 nodes 向量，
			// 以保证每对节点可以成为一段线段。这里的 nodes 表示 Dexel 中存储的节点信息
			if (nodes.size() % 2 != 0)
				nodes.clear();
		}
		
		// 这个函数用于将存储在 Dexel 中的节点信息转换为一组线段（Segment）。
		auto segments() const -> std::vector<Segment> {
			// 进行断言，确保节点数量为偶数。因为每两个节点会构成一个线段，所以节点数量应该是偶数，如果不满足这个条件，就会触发断言失败。
			assert(nodes.size() % 2 == 0);
			// 创建一个名为 segs 的空的 std::vector<Segment> 向量，用于存储转换后的线段
			std::vector<Segment> segs;
			// 预分配足够的空间，以避免多次重新分配内存。由于每两个节点构成一个线段，所以向量的大小可以设为节点数量的一半
			segs.reserve(nodes.size() / 2);
			// 使用循环遍历 nodes 向量中的每两个相邻节点，将它们作为起始节点和终止节点创建一个 Segment 对象，并将该对象添加到 segs 向量中。
			for (size_t i = 0; i < nodes.size(); i += 2)
				segs.push_back({nodes[i], nodes[i + 1]});
			// 循环结束后，将存储了转换后的线段的 segs 向量返回
			return segs;
		}

		// 存储 Node 结构体对象的动态数组（向量）。
		std::vector<Node> nodes;
	};
	// 定义了一个名为 DexelImage 的结构体，表示一个 Dexel 图像。Dexel 是 TriDexel 算法中的一个概念，用于存储体素的信息。
	// 这个结构体定义了用于访问和操作 Dexel 图像数据的方法。
	struct DexelImage {
		// 表示当前 Dexel 图像所在的坐标轴索引
		unsigned int axis0;
		// 表示另外两个坐标轴上的分辨率
		unsigned int axis1Res;
		unsigned int axis2Res;
		std::vector<Dexel> dexels;

		auto dexel(unsigned int axis1, unsigned int axis2) -> Dexel & {
			return dexels[axis1Res * axis1 + axis2];
		}

		auto dexel(unsigned int axis1, unsigned int axis2) const -> const Dexel & {
			return dexels[axis1Res * axis1 + axis2];
		}
	};
	// 定义了一个名为 TriDexelImage 的类，表示一个 TriDexel 图像。TriDexel 是一种体素化算法，用于将三维空间划分为体素，用于表示物体或场景的几何信息。
	class TriDexelImage {
	public:
		// 构造函数 TriDexelImage(uvec3 resolution)：接受一个 uvec3 类型的参数 resolution，用于指定图像在每个维度上的分辨率。在构造函数中，
		// 通过循环来初始化三个 DexelImage 对象，分别对应三个不同的坐标轴。DexelImage 是另一个类型，用于表示一个 Dexel 图像。

		// : resolution(resolution) 部分是成员初始化列表，它指定了成员变量 resolution 的初始化方式。通过这种方式，在构造函数的开始阶段就可以将参数
		//  resolution 的值赋给成员变量 resolution，而不是在构造函数的函数体中进行赋值操作。
		TriDexelImage(uvec3 resolution)
			: resolution(resolution) {
			for (auto axis0 : {0, 1, 2}) {
				const auto axis1 = (axis0 + 1) % 3;
				const auto axis2 = (axis0 + 2) % 3;
				images[axis0].axis0 = axis0;
				images[axis0].axis1Res = resolution[axis1];
				images[axis0].axis2Res = resolution[axis2];
				images[axis0].dexels.resize(resolution[axis1] * resolution[axis2]);
			}
		}

		std::array<DexelImage, 3> images;
		uvec3 resolution;
	};

	struct boool {
		bool value;

		auto operator=(bool b) -> boool & {
			value = b;
			return *this;
		}

		operator bool() const {
			return value;
		}
	};

	struct Cell {
		BoundingBox box;
		std::array<bool, 8> occupancies;
		std::array<std::vector<Segment>, 12> edges;

		auto realPoint(unsigned int i) const -> vec3 {
			const auto& u = box.upper;
			const auto& l = box.lower;
			switch (i) {
			case 0: return {l.x, l.y, l.z};
			case 1: return {u.x, l.y, l.z};
			case 2: return {l.x, u.y, l.z};
			case 3: return {u.x, u.y, l.z};
			case 4: return {l.x, l.y, u.z};
			case 5: return {u.x, l.y, u.z};
			case 6: return {l.x, u.y, u.z};
			case 7: return {u.x, u.y, u.z};
			default: std::terminate();
			}
		}

		auto empty() const {
			return std::all_of(begin(occupancies), end(occupancies), [](bool o) { return o == false; });
		}
	};

	constexpr auto edgeIndexOffsets = std::array<uvec3, 12>{ {
		{0, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
		{0, 1, 1},
		{0, 0, 0},
		{1, 0, 0},
		{0, 0, 1},
		{1, 0, 1},
		{0, 0, 0},
		{1, 0, 0},
		{0, 1, 0},
		{1, 1, 0}
	}};

	auto gridToDepth(unsigned int coord, unsigned int axis, uvec3 res, BoundingBox box) -> float {
		return (coord / (float)(res[axis] - 1)) * (box.upper[axis] - box.lower[axis]) + box.lower[axis];
	}

	auto depthToGrid(float depth, unsigned int axis, uvec3 res, BoundingBox box) -> unsigned {
		return static_cast<unsigned>((depth - box.lower[axis]) / (box.upper[axis] - box.lower[axis]) * (res[axis] - 1));
	}

	constexpr std::array<std::array<unsigned int, 2>, 12> edgeToPointIds = {{
		{{0, 1}},
		{{2, 3}},
		{{4, 5}},
		{{6, 7}},
		{{0, 2}},
		{{1, 3}},
		{{4, 6}},
		{{5, 7}},
		{{0, 4}},
		{{1, 5}},
		{{2, 6}},
		{{3, 7}},
	}};

	constexpr auto edgeToAxis = []() constexpr {
		std::array<std::array<unsigned int, 8>, 8> t{};
		t[0][1] = t[1][0] = 0;
		t[2][3] = t[3][2] = 0;
		t[4][5] = t[5][4] = 0;
		t[6][7] = t[7][6] = 0;
		t[0][2] = t[2][0] = 1;
		t[1][3] = t[3][1] = 1;
		t[4][6] = t[6][4] = 1;
		t[5][7] = t[7][5] = 1;
		t[0][4] = t[4][0] = 2;
		t[1][5] = t[5][1] = 2;
		t[2][6] = t[6][2] = 2;
		t[3][7] = t[7][3] = 2;
		return t;
	}();

	auto clampSegments(std::vector<Segment> segments, float min, float max) {
		for (auto& seg : segments) {
			seg.start.depth = std::clamp(seg.start.depth, min, max);
			seg.end.depth = std::clamp(seg.end.depth, min, max);
		}

		segments.erase(std::remove_if(begin(segments), end(segments), [](const Segment& seg) {
			return seg.start.depth == seg.end.depth;
		}), end(segments));

		return segments;
	}

	class TriDexelGrid {
	public:
		TriDexelGrid(TriDexelImage& image, BoundingBox box)
			: image(image), box(box) {
			const auto r = image.resolution + 1u;
			occupancies.resize(r.x * r.y * r.z);
		}

		auto& occupancy(uvec3 index) {
			const auto& r = image.resolution;
			return occupancies[index.z * (r.x * r.y) + index.y * r.x + index.x];
		}

		auto cell(uvec3 index) -> Cell {
			Cell c;
			for (auto axis : {0, 1, 2}) {
				c.box.lower[axis] = gridToDepth(index[axis] + 0, axis, image.resolution, box);
				c.box.upper[axis] = gridToDepth(index[axis] + 1, axis, image.resolution, box);
			}

			c.occupancies[0] = occupancy(index + uvec3{0, 0, 0});
			c.occupancies[1] = occupancy(index + uvec3{1, 0, 0});
			c.occupancies[2] = occupancy(index + uvec3{0, 1, 0});
			c.occupancies[3] = occupancy(index + uvec3{1, 1, 0});
			c.occupancies[4] = occupancy(index + uvec3{0, 0, 1});
			c.occupancies[5] = occupancy(index + uvec3{1, 0, 1});
			c.occupancies[6] = occupancy(index + uvec3{0, 1, 1});
			c.occupancies[7] = occupancy(index + uvec3{1, 1, 1});

			for (auto i = 0; i < 12; i++) {
				const auto [src, dst] = edgeToPointIds[i];
				const auto axis0 = edgeToAxis[src][dst];
				const auto axis1 = (axis0 + 1) % 3;
				const auto axis2 = (axis0 + 2) % 3;
				const auto edgeIndex = index + edgeIndexOffsets[i];
				c.edges[i] = clampSegments(image.images[axis0].dexel(edgeIndex[axis1], edgeIndex[axis2]).segments(), c.box.lower[axis0], c.box.upper[axis0]);
			}

			return c;
		}

		TriDexelImage& image;
		BoundingBox box;

		std::vector<boool> occupancies;
	};

	// 定义了一个函数 uniformResolution，用于计算在给定的包围盒内使用均匀分辨率的体素的数量。
	auto uniformResolution(BoundingBox box, unsigned int resolution) {
		// 计算包围盒的每个维度的尺寸（大小），即 boxSizes，这是通过将上界减去下界得到的。
		const auto boxSizes = box.upper - box.lower;
		// 从 boxSizes 中找到最大的尺寸值，表示在包围盒的三个维度中，最大的尺寸值。这个最大的尺寸值用于确保在所有维度上都有足够的均匀分辨率体素。
		const auto largest = std::max({boxSizes.x, boxSizes.y, boxSizes.z});
		// 计算每个体素的大小 cellSize，即将最大尺寸值除以分辨率
		const auto cellSize = largest / resolution;
		// 使用 glm::round 函数来计算每个维度上的体素数量，通过将包围盒尺寸除以体素大小，并进行四舍五入以确保得到整数值。
		// 返回一个 uvec3 类型的向量，表示每个维度上的体素数量。
		return uvec3{glm::round(boxSizes / cellSize)};
	}

	auto createRay(BoundingBox box, unsigned int axis0, unsigned int axis1, unsigned int axis2, float deltaX, float deltaY, unsigned int x, unsigned int y) {
		auto origin = box.lower;
		origin[axis1] += x * deltaX;
		origin[axis2] += y * deltaY;
		auto direction = vec3{};
		direction[axis0] = 1;
		return Ray{origin, direction};
	}

	template<typename HitFunc>
	void axisParallelRaycast(BoundingBox box, uvec3 res, RaycastCallback rcc, HitFunc hitFunc) {
		for (auto axis0 : {0, 1, 2}) {
			const auto axis1 = (axis0 + 1) % 3;
			const auto axis2 = (axis0 + 2) % 3;
			const auto xCount = res[axis1];
			const auto yCount = res[axis2];
			const auto deltaX = (box.upper[axis1] - box.lower[axis1]) / (xCount - 1);
			const auto deltaY = (box.upper[axis2] - box.lower[axis2]) / (yCount - 1);
			for (auto y = 0u; y < yCount; y++) {
				for (auto x = 0u; x < xCount; x++) {
					const auto ray = createRay(box, axis0, axis1, axis2, deltaX, deltaY, x, y);
					rcc(ray, [&](float depth, vec3 normal) {
						hitFunc(axis0, x, y, depth, normal);
						});
				}
			}
		}
	}

	void assignOccupancies(TriDexelGrid& grid) {
		for (auto& image : grid.image.images) {
			const auto axis0 = image.axis0;
			const auto axis1 = (axis0 + 1) % 3;
			const auto axis2 = (axis0 + 2) % 3;
			for (auto axis2Val = 0u; axis2Val < image.axis2Res; axis2Val++) {
				for (auto axis1Val = 0u; axis1Val < image.axis1Res; axis1Val++) {
					auto& dexel = image.dexels[axis2Val * image.axis1Res + axis1Val];
					dexel.regularize();
					for (const auto& seg : dexel.segments()) {
						const auto start = std::max(0u, depthToGrid(seg.start.depth, image.axis0, grid.image.resolution, grid.box));
						const auto end = std::min(grid.image.resolution[axis0] - 1, depthToGrid(seg.end.depth, image.axis0, grid.image.resolution, grid.box) + 1);
						for (auto axis0Val = start; axis0Val <= end; axis0Val++) {
							const auto exactAxis0Depth = gridToDepth(axis0Val, axis0, grid.image.resolution, grid.box);
							if (seg.start.depth > exactAxis0Depth || seg.end.depth < exactAxis0Depth)
								continue;
							uvec3 index;
							index[axis0] = axis0Val;
							index[axis1] = axis1Val;
							index[axis2] = axis2Val;
							grid.occupancy(index) = true;
						}
					}
				}
			}
		}
	}

	constexpr auto rho = 1e-1f;

	void regularizeCell(Cell& cell) {
		for (auto i = 0; i < 12; i++) {
			auto& segs = cell.edges[i];
			const auto [src, dst] = edgeToPointIds[i];
			const auto axis = i / 4;
			const auto srcDepth = cell.box.lower[axis];
			const auto dstDepth = cell.box.upper[axis];
			const auto srcOcc = cell.occupancies[src];
			const auto dstOcc = cell.occupancies[dst];

			// rule 1
			if (srcOcc && dstOcc) {
				segs.clear();
				segs.push_back({{srcDepth, {}}, {dstDepth, {}}});
			}

			// rule 2
			if (!srcOcc && !dstOcc) {
				segs.clear();
			}

			// rule 3
			glm::vec3 n{0, 0, 0};
			n[axis] = 1;
			if (srcOcc && !dstOcc) {
				if (segs.size() == 0 || segs.front().start.depth - rho > srcDepth)
					segs.push_back({{srcDepth, -n}, {srcDepth + rho, n}});
				else if (segs.size() > 0 && segs.front().start.depth > srcDepth && segs.front().start.depth - rho <= srcDepth)
					segs.front().start.depth = srcDepth;
			}
			if (!srcOcc && dstOcc) {
				if (segs.size() == 0 || segs.back().end.depth + rho < dstDepth)
					segs.push_back({{dstDepth - rho, -n}, {dstDepth, n}});
				else if (segs.size() > 0 && segs.back().end.depth < dstDepth && segs.back().end.depth + rho >= dstDepth)
					segs.back().end.depth = dstDepth;
			}

			// rule 4
			segs.erase(std::remove_if(begin(segs), end(segs), [&](const Segment& s) {
				return s.start.depth != srcDepth && s.end.depth != dstDepth;
				}), end(segs));
		}
	}

	auto occupancyToCase(const std::array<bool, 8> & occupancies) -> unsigned int {
		unsigned int c = 0;
		for (auto i = 0u; i < 8; i++)
			if (occupancies[i])
				c |= 1u << i;
		return c;
	}

	auto isProblematicCase(const Cell& cell) {
		switch (occupancyToCase(cell.occupancies)) {
		case 24:
		case 36:
		case 66:
		case 129:
			return true;
		default:
			return false;
		}
	}

	struct EdgeVertex {
		unsigned int point;
		unsigned int neighbor;
		vec3 position;
		vec3 normal;
	};

	const auto pointsToEdgeId = []() {
		std::array<std::array<unsigned int, 8>, 8> t{};
		for (auto& tt : t)
			tt.fill(std::numeric_limits<unsigned int>::max());
		t[0][1] = t[1][0] = 0;
		t[2][3] = t[3][2] = 1;
		t[4][5] = t[5][4] = 2;
		t[6][7] = t[7][6] = 3;
		t[0][2] = t[2][0] = 4;
		t[1][3] = t[3][1] = 5;
		t[4][6] = t[6][4] = 6;
		t[5][7] = t[7][5] = 7;
		t[0][4] = t[4][0] = 8;
		t[1][5] = t[5][1] = 9;
		t[2][6] = t[6][2] = 10;
		t[3][7] = t[7][3] = 11;
		return t;
	}();

	constexpr std::array<std::array<unsigned int, 3>, 8> neighborIds = {{
		{{1, 4, 2}},
		{{0, 3, 5}},
		{{0, 6, 3}},
		{{1, 2, 7}},
		{{0, 5, 6}},
		{{1, 7, 4}},
		{{2, 4, 7}},
		{{3, 6, 5}},
	}};

	auto depthFirstSearch(bool occupied, unsigned int cur, unsigned int last, const Cell& cell, std::array<bool, 8> & visited, std::vector<EdgeVertex>& loop) {
		if (cell.occupancies[cur] != occupied || visited[cur])
			return;
		visited[cur] = true;
		auto neighbors = neighborIds[cur];
		const auto lastIndex = std::distance(begin(neighbors), std::find(begin(neighbors), end(neighbors), last));
		std::rotate(begin(neighbors), begin(neighbors) + (lastIndex + 1) % 3, end(neighbors));

		for (const auto n : neighbors) {
			if (cell.occupancies[n] == occupied) {
				depthFirstSearch(occupied, n, cur, cell, visited, loop);
				continue;
			}

			const auto axis = edgeToAxis[cur][n];
			const auto& seg = cell.edges[pointsToEdgeId[cur][n]].front();
			const auto curReal = cell.realPoint(cur);
			const auto nReal = cell.realPoint(n);
			Node node;
			if (seg.start.depth == curReal[axis] || seg.start.depth == nReal[axis])
				node = seg.end;
			else
				node = seg.start;
			auto v = curReal;
			v[axis] = node.depth;
			loop.push_back({cur, n, v, node.normal});
		}
	}

	auto findLoops(const Cell& cell, bool occupied) {
		std::array<bool, 8> visited{};
		std::vector<std::vector<EdgeVertex>> loops;
		for (auto start = 0u; start < 8; start++) {
			std::vector<EdgeVertex> loop;
			depthFirstSearch(occupied, start, 0, cell, visited, loop);
			if (!loop.empty())
				loops.push_back(std::move(loop));
		}
		return loops;
	}

	void triangulateLoopIntoFan(const std::vector<vec3>& loop, vec3 center, std::vector<Triangle>& triangles) {
		if (loop.size() < 3)
			return;
		for (auto a = 0; a < loop.size(); a++) {
			auto b = (a + 1) % loop.size();
			triangles.push_back({center, loop[a], loop[b]});
		}
	}

	void triangulateLoopIntoFan(const std::vector<vec3>& loop, std::vector<Triangle>& triangles) {
		const auto center = std::accumulate(begin(loop), end(loop), vec3{0, 0, 0}) / static_cast<float>(loop.size());
		return triangulateLoopIntoFan(loop, center, triangles);
	}

	void triangulateLoopAtFirst(const std::vector<vec3>& loop, std::vector<Triangle>& triangles) {
		if (loop.size() < 3)
			return;
		const auto center = loop.front();
		for (auto b = 2; b < loop.size() - 1; b++) {
			const auto a = b - 1;
			triangles.push_back({center, loop[a], loop[b]});
		}
	}

	constexpr auto planeEpsilon = 1e-6f;

	auto intersectPlanes(Plane a, Plane b, Plane c) -> std::optional<vec3> {
		const auto det = glm::dot(glm::cross(a.n, b.n), c.n);
		if (std::abs(det) < planeEpsilon)
			return {};
		return (glm::cross(b.n, c.n) * a.d + glm::cross(c.n, a.n) * b.d + glm::cross(a.n, b.n) * c.d) / det;
	}

	auto pointLineDistance(vec3 a, vec3 b, vec3 p) {
		return glm::length(glm::cross(p - a, p - b)) / glm::length(a - b);
	}

	template <typename T>
	void removeCyclicRange(std::vector<T>& loop, int first, int last) {
		if (first <= last)
			loop.erase(begin(loop) + first, begin(loop) + last + 1);
		else {
			loop.erase(begin(loop), begin(loop) + last + 1);
			loop.erase(begin(loop) + first - last, end(loop));
		}
	}

	const auto cellSideNormals = []() {
		std::array<std::array<std::array<vec3, 8>, 8>, 8> t{};

		auto assign3of4 = [&](int a, int b, int c, int d, vec3 n) {
			t[a][b][c] = n;
			t[a][b][d] = n;
			t[a][c][d] = n;
			t[b][c][d] = n;
		};
		assign3of4(0, 2, 4, 6, {-1,  0,  0});
		assign3of4(1, 3, 5, 7, {1,  0,  0});
		assign3of4(0, 1, 4, 5, {0, -1,  0});
		assign3of4(2, 3, 6, 7, {0,  1,  0});
		assign3of4(0, 1, 2, 3, {0,  0, -1});
		assign3of4(4, 5, 6, 7, {0,  0,  1});
		return t;
	}();

	constexpr auto lineEpsilon = 1e-6f;

	void triangulateLoopRefined(const std::vector<EdgeVertex>& loop, const Cell& cell, std::vector<Triangle>& triangles) {
		const auto inside = [&](vec3 p) {
			return
				cell.box.lower.x < p.x && p.x < cell.box.upper.x &&
				cell.box.lower.y < p.y && p.y < cell.box.upper.y &&
				cell.box.lower.z < p.z && p.z < cell.box.upper.z;
		};
		std::vector<const EdgeVertex*> apexVertices;
		std::vector<vec3> finalLoop;
		std::vector<bool> isEdgeVertex;
		auto intermedCount = 0;

		// create new boundary loop with intermediate vertices
		for (auto i = 0; i < loop.size(); i++) {
			const auto& a = loop[i];
			const auto& b = loop[(i + 1) % loop.size()];
			finalLoop.push_back(a.position);
			isEdgeVertex.push_back(true);
			std::vector<unsigned int> points{a.point, a.neighbor, b.point, b.neighbor};
			std::sort(begin(points), end(points));
			points.erase(std::unique(begin(points), end(points)), end(points));
			assert(points.size() >= 3);
			auto n = cellSideNormals[points[0]][points[1]][points[2]];
			auto cellPlane = Plane{n, cell.realPoint(points[0])};
			auto aPlane = Plane{a.normal, a.position};
			auto bPlane = Plane{b.normal, b.position};
			auto intermed = intersectPlanes(cellPlane, aPlane, bPlane);
			if (intermed) {
				const auto cellPlaneAxis = std::fabs(n[0]) > 0 ? 0 : (std::fabs(n[1]) > 0 ? 1 : 2);
				(*intermed)[cellPlaneAxis] = cell.realPoint(points[0])[cellPlaneAxis];
				if (inside(*intermed)) {
					const auto dist = pointLineDistance(a.position, b.position, *intermed);
					if (dist > lineEpsilon) {
						finalLoop.push_back(*intermed);
						isEdgeVertex.push_back(false);
						intermedCount++;
						apexVertices.push_back(&a);
						apexVertices.push_back(&b);
					}
				}
			}
		}

		// try to create an apex vertex
		std::optional<vec3> apex;
		if (intermedCount >= 3) {
			std::sort(begin(apexVertices), end(apexVertices));
			apexVertices.erase(std::unique(begin(apexVertices), end(apexVertices)), end(apexVertices));
			assert(apexVertices.size() >= 3);
			std::vector<Plane> planes;
			for (auto i : {0, 1, 2})
				planes.push_back({apexVertices[i]->normal, apexVertices[i]->position});
			const auto a = intersectPlanes(planes[0], planes[1], planes[2]);
			if (a && inside(*a))
				apex = a;
		}

		if (apex)
			triangulateLoopIntoFan(finalLoop, *apex, triangles);
		else if (intermedCount == 0)
			triangulateLoopIntoFan(finalLoop, triangles);
		else if (intermedCount == 1) {
			std::rotate(begin(finalLoop), begin(finalLoop) + std::distance(begin(isEdgeVertex), std::find(begin(isEdgeVertex), end(isEdgeVertex), false)), end(finalLoop));
			triangulateLoopAtFirst(finalLoop, triangles);
		} else {
			for (auto first = 0u; first < finalLoop.size(); first++) {
				if (isEdgeVertex[first])
					continue;
				auto last = (first + 1) % (unsigned int)finalLoop.size();
				while (isEdgeVertex[last])
					last = (last + 1) % (unsigned int)finalLoop.size();
				std::vector<vec3> subLoop;
				for (auto i = first; i != last; i = (i + 1) % (unsigned int)finalLoop.size())
					subLoop.push_back(finalLoop[i]);
				subLoop.push_back(finalLoop[last]);
				assert(subLoop.size() >= 3);
				triangulateLoopIntoFan(subLoop, triangles);
				removeCyclicRange(finalLoop, first + 1, last - 1);
				removeCyclicRange(isEdgeVertex, first + 1, last - 1);
			}
			triangulateLoopIntoFan(finalLoop, triangles);
		}
	}

	auto reverseLoops(std::vector<std::vector<EdgeVertex>> loops) {
		for (auto& loop : loops)
			std::reverse(begin(loop), end(loop));
		return loops;
	}

	void triangulateCell(const Cell& cell, std::vector<Triangle>& triangles) {
		std::vector<std::vector<EdgeVertex>> loops;
		if (!isProblematicCase(cell))
			loops = findLoops(cell, false);
		else
			loops = reverseLoops(findLoops(cell, true));
		for (const auto& loop : loops)
			triangulateLoopRefined(loop, cell, triangles);
	}

	void dumpDexels(const TriDexelGrid& grid, std::filesystem::path path) {
		std::vector<Triangle> result;

		for (auto axis0 : {0, 1, 2}) {
			const auto& img = grid.image.images[axis0];
			const auto axis1 = (axis0 + 1) % 3;
			const auto axis2 = (axis0 + 2) % 3;
			const auto xCount = img.axis1Res;
			const auto yCount = img.axis2Res;
			const auto deltaX = (grid.box.upper[axis1] - grid.box.lower[axis1]) / (xCount - 1);
			const auto deltaY = (grid.box.upper[axis2] - grid.box.lower[axis2]) / (yCount - 1);

			for (auto y = 0u; y < yCount; y++) {
				for (auto x = 0u; x < xCount; x++) {
					const auto origin = createRay(grid.box, axis0, axis1, axis2, deltaX, deltaY, x, y).origin;

					const auto& dexel = img.dexel(x, y);
					for (const auto& s : dexel.segments()) {
						auto start = origin;
						start[axis0] = s.start.depth;
						auto end = origin;
						end[axis0] = s.end.depth;
						result.push_back({start, (start + end) / 2.0f, end});
					}
				}
			}
		}

		saveTriangles(path, result);
	}

	void dumpCell(const Cell& cell, std::filesystem::path path) {
		savePoints(path / "box.ply", {cell.box.lower, cell.box.upper});

		std::vector<vec3> realPoints;
		for (auto i = 0; i < 8; i++)
			realPoints.push_back(cell.realPoint(i));
		savePoints(path / "realPoints.ply", realPoints);

		std::vector<Triangle> edges;
		for (auto i = 0; i < 12; i++) {
			const auto [src, dst] = edgeToPointIds[i];
			const auto axis = i / 4;
			const auto p = cell.realPoint(src);
			for (const auto& e : cell.edges[i]) {
				vec3 start = p;
				start[axis] = e.start.depth;
				vec3 end = p;
				end[axis] = e.end.depth;
				edges.push_back({start, (start + end) / 2.0f, end});
			}
		}
		saveTriangles(path / "edges.stl", edges);
	}
}

// 这个函数的作用是进行 TriDexel 算法，这是一种用于生成三维体素表示的算法，通常在体素化、体素渲染等领域中应用。
// 輸入参数1.包围盒box，2.resolution分辨率，3.回调函数RaycastCallback rcc
// 返回一个包含三角形的向量 std::vector<Triangle>    Triangle这个模板在头文件tridexel.h中有详细说明
auto tridexel(BoundingBox box, unsigned int resolution, RaycastCallback rcc) -> std::vector<Triangle> {

	// res 是通过调用函数 uniformResolution 来计算得到的一个变量。它表示在均匀分辨率下，每个维度上的体素数量
	const auto res = uniformResolution(box, resolution);

	auto img = TriDexelImage{ res };
	axisParallelRaycast(box, res, rcc, [&](unsigned int axis, unsigned int x, unsigned int y, float depth, vec3 normal) {
		auto& im = img.images[axis];
		im.dexel(x, y).nodes.push_back({depth, normal});
	});

	auto grid = TriDexelGrid{ img, box };
	assignOccupancies(grid);

	//dumpDexels(grid, "dexels.stl");

	std::vector<Triangle> triangles;
	for (auto z = 0u; z < grid.image.resolution.z - 1; z++) {
		for (auto y = 0u; y < grid.image.resolution.y - 1; y++) {
			for (auto x = 0u; x < grid.image.resolution.x - 1; x++) {
				auto cell = grid.cell({ x, y, z });
				if (!cell.empty()) {
					//const auto folder = "cell_" + std::to_string(x) + "_" + std::to_string(y) + "_" + std::to_string(z);
					//dumpCell(cell, folder + "_before");
					regularizeCell(cell);
					//dumpCell(cell, folder + "_after");

					std::vector<Triangle> cellTriangles;
					triangulateCell(cell, cellTriangles);
					//saveTriangles(folder + "_after/triangles.stl", cellTriangles);

					triangles.insert(end(triangles), begin(cellTriangles), end(cellTriangles));
				}
			}
		}
	}

	return triangles;
}
