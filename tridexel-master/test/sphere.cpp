// Boost C++ 库中的一个头文件，用于包含静态向量（static vector）的相关定义和功能。 C++ 编程的工具和组件。
#include <boost/container/static_vector.hpp>
// 包含 GTest 框架头文件的指令，用于引入测试框架的定义和功能。
#include <gtest/gtest.h>
// GLM（OpenGL Mathematics）库的主要头文件，用于包含 GLM 库中的基本数学类型、函数和操作。
#include <glm/glm.hpp>
// <optional> 是 C++17 中引入的标准头文件，用于包含 std::optional 类的定义。std::optional 是一个表示可能存在或不存在值的类模板，它可以用于在可能缺少值的情况下进行更加安全和灵活的编程。
#include <optional>

#include <fstream>


#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <limits>
#include <stack> 


#include "../src/tridexel.h"
#include "../src/IO.h"

#include<stdio.h>
#include<stdlib.h>
#include<string.h>


//文件头，共84字节
struct Head
{
	char partName[80];//零件名称
	int  faceNum;//面的数目
};
 
//点，三个float类型的，大小为12字节
struct Point
{
	float x;
	float y;
	float z;
};
 
//法线
struct Normal
{
	float i;
	float j;
	float k;
};
 


// Ray 结构体表示一个射线，具有两个成员变量
struct RayTrans {
	// origin：表示射线的起始点，是一个 glm::vec3 类型的向量，存储了射线的起始坐标。
	glm::vec3 origin;
	// direction：表示射线的方向，也是一个 glm::vec3 类型的向量，存储了射线的方向向量。
	glm::vec3 direction;
};




namespace STLTrans{
	struct Triangle
	{
		glm::vec3 vertex1;//三角形的顶点一
		glm::vec3 vertex2;//三角形的顶点二
		glm::vec3 vertex3;//三角形的顶点三
		glm::vec3 normal;//三角形的的法向量
		glm::vec3 center;
		float radius;
		char info[2]; // 保留数据，可以不用管

		bool intersect(const RayTrans &ray, float &t) const
		{
			glm::vec3 edge1 = vertex2 - vertex1;
			glm::vec3 edge2 = vertex3 - vertex1;
			glm::vec3 h = glm::cross(ray.direction, edge2);
			float a = glm::dot(edge1, h);

			if (a > -std::numeric_limits<float>::epsilon() && a < std::numeric_limits<float>::epsilon())
				return false;

			float f = 1.0f / a;
			glm::vec3 s = ray.origin - vertex1;
			float u = f * glm::dot(s, h);

			if (u < 0.0f || u > 1.0f)
				return false;

			glm::vec3 q = glm::cross(s, edge1);
			float v = f * glm::dot(ray.direction, q);

			if (v < 0.0f || u + v > 1.0f)
				return false;

			t = f * glm::dot(edge2, q);

			return t > std::numeric_limits<float>::epsilon();
		}
	};

};

namespace Calculate
{
	void calculateCircumcircle(STLTrans::Triangle &triangle)
	{
		// 计算三角形的中心点
		triangle.center = (triangle.vertex1 + triangle.vertex2 + triangle.vertex3) / 3.0f;

		// 计算每个顶点到中心点的距离并找到最大的距离
		float dist1 = glm::length(triangle.vertex1 - triangle.center);
		float dist2 = glm::length(triangle.vertex2 - triangle.center);
		float dist3 = glm::length(triangle.vertex3 - triangle.center);

		// 使用最大距离作为外接圆的半径
		triangle.radius = std::max(dist1, std::max(dist2, dist3));
	}

	void CalculateBox(const std::vector<STLTrans::Triangle> &triangles, float &minX, float &minY, float &minZ, float &maxX, float &maxY, float &maxZ)
	{
		if (triangles.empty())
		{
			// 如果三角形向量为空，将最小值设为FLT_MAX，最大值设为-FLT_MAX
			minX = FLT_MAX;
			minY = FLT_MAX;
			minZ = FLT_MAX;
			maxX = -FLT_MAX;
			maxY = -FLT_MAX;
			maxZ = -FLT_MAX;
			return;
		}

		for (const STLTrans::Triangle &triangle : triangles)
		{
			// 遍历三角形，更新最小和最大坐标值
			for (int i = 0; i < 3; i++)
			{
				const glm::vec3 &vertex = (i == 0) ? triangle.vertex1 : ((i == 1) ? triangle.vertex2 : triangle.vertex3);
				minX = std::min(minX, vertex.x);
				minY = std::min(minY, vertex.y);
				minZ = std::min(minZ, vertex.z);
				maxX = std::max(maxX, vertex.x);
				maxY = std::max(maxY, vertex.y);
				maxZ = std::max(maxZ, vertex.z);
			}
		}
	}

	// BVH节点数据结构
	struct BVHNode {
		BVHNode* left;
		BVHNode* right;
		BoundingBox boundingBox; // 包围盒
		std::vector<int> triangleIndices; // 三角形索引
	};

	BoundingBox CalculateAABB(const STLTrans::Triangle& triangle) {
		BoundingBox boundingBox;
		boundingBox.lower = glm::min(glm::min(triangle.vertex1, triangle.vertex2), triangle.vertex3);
		boundingBox.upper = glm::max(glm::max(triangle.vertex1, triangle.vertex2), triangle.vertex3);
		return boundingBox;
	}

	// 合并两个包围盒
	BoundingBox MergeAABBs(const BoundingBox& box1, const BoundingBox& box2) {
		BoundingBox mergedBox;
		mergedBox.lower = glm::min(box1.lower, box2.lower);
		mergedBox.upper = glm::max(box1.upper, box2.upper);
		return mergedBox;
	}

	// 构建BVH树
	// BVHNode* BuildBVHTree(const std::vector<STLTrans::Triangle>& triangles, int start, int end) {
	// 	if (start == end) {
	// 		// 当前节点为叶子节点
	// 		BVHNode* leafNode = new BVHNode;
	// 		leafNode->left = nullptr;
	// 		leafNode->right = nullptr;
	// 		leafNode->boundingBox = CalculateAABB(triangles[start]);
	// 		leafNode->triangleIndices.push_back(start);
	// 		return leafNode;
	// 	} else {
	// 		// 当前节点为内部节点
	// 		BVHNode* internalNode = new BVHNode;
	// 		internalNode->left = BuildBVHTree(triangles, start, (start + end) / 2);
	// 		internalNode->right = BuildBVHTree(triangles, (start + end) / 2 + 1, end);
	// 		internalNode->boundingBox = MergeAABBs(internalNode->left->boundingBox, internalNode->right->boundingBox);
	// 		return internalNode;
	// 	}
	// }







Calculate::BVHNode* BuildBVHTree(const std::vector<STLTrans::Triangle>& triangles, int start, int end) {
    if (start == end) {
        // 当前节点为叶子节点
        BVHNode* leafNode = new BVHNode;
        leafNode->left = nullptr;
        leafNode->right = nullptr;
        leafNode->boundingBox = CalculateAABB(triangles[start]);
        leafNode->triangleIndices.push_back(start);

        // std::cout << "Triangle index " << start << " included in BVH tree.\n";

        return leafNode;
    } else {
        // 当前节点为内部节点
        BVHNode* internalNode = new BVHNode;
        internalNode->left = BuildBVHTree(triangles, start, (start + end) / 2);
        internalNode->right = BuildBVHTree(triangles, (start + end) / 2 + 1, end);
        internalNode->boundingBox = MergeAABBs(internalNode->left->boundingBox, internalNode->right->boundingBox);

        // std::cout << "Internal node: Triangles " << start << " to " << end << " included in BVH tree.\n";

        return internalNode;
    }
}










	
	bool boundingBoxIntersect(const BoundingBox& bbox, const RayTrans& ray) {
		glm::vec3 invDir = 1.0f / ray.direction;
		glm::vec3 t1 = (bbox.lower - ray.origin) * invDir;
		glm::vec3 t2 = (bbox.upper - ray.origin) * invDir;

		glm::vec3 tmin = glm::min(t1, t2);
		glm::vec3 tmax = glm::max(t1, t2);

		float tminmax = glm::max(glm::max(tmin.x, tmin.y), tmin.z);
		float tmaxmin = glm::min(glm::min(tmax.x, tmax.y), tmax.z);

		return tminmax <= tmaxmin;
	}

	void IntersectBVHNode(const BVHNode* node, const RayTrans& ray, const std::vector<STLTrans::Triangle>& triangles, std::vector<glm::vec3>& intersectionPoints) {
		if (!boundingBoxIntersect(node->boundingBox, ray))
			return; // 如果射线不与节点的包围盒相交，则不相交
		
		if (node->left == nullptr && node->right == nullptr) {
			// 这是叶子节点，即没有左子树和右子树
			// 检查射线与节点包含的三角形相交
			for (int triangleIndex : node->triangleIndices) {
				const STLTrans::Triangle& triangle = triangles[triangleIndex];
				float t;
				if (triangle.intersect(ray, t)) {
					// 计算交点坐标并添加到结果集中
					glm::vec3 intersectionPoint = ray.origin + ray.direction * t;
					intersectionPoints.push_back(intersectionPoint);
					// 打印交点坐标
    // std::cout << "Intersection point: (" << intersectionPoint.x << ", " << intersectionPoint.y << ", " << intersectionPoint.z << ")\n";
				}
			}
		} else {
			// 递归检查左子树和右子树
			IntersectBVHNode(node->left, ray, triangles, intersectionPoints);
			IntersectBVHNode(node->right, ray, triangles, intersectionPoints);
		}
	}

	void DeleteBVHTree(BVHNode* node) {
		if (node == nullptr) {
			return;
		}

		// 递归删除左子树和右子树
		DeleteBVHTree(node->left);
		DeleteBVHTree(node->right);

		// 释放当前节点
		delete node;
	}

	

}


namespace {

	// 三维向量
	using glm::vec3;
	// 无符号三维向量
	using glm::uvec3;

	struct Segment {
		glm::vec3 point1;
		glm::vec3 point2;

		Segment(glm::vec3 p1, glm::vec3 p2) : point1(p1), point2(p2) {}
	};

	
	// 表示一个平面。它有两个构造函数，分别用于根据法线和距离、以及法线和一个点来初始化平面。
	struct Plane {
		Plane(vec3 n, float d) : n(n), d(d) {}
		Plane(vec3 n, vec3 p) : n(n), d(glm::dot(n, p)) {}

		// vec3 n: 表示平面的法线向量。
		// float d: 表示平面距离原点的距离。
		vec3 n;
		float d;
	};

	// 结构体对象Sphere（球体）
	struct Sphere {
		// center 是一个 glm::vec3 类型的变量，表示球体的中心点坐标，
		// glm::vec3 是 glm 库中的一个类型，表示三维向量。
		glm::vec3 center;
		// radius 是一个 float 类型的变量，表示球体的半径。
		float radius;
	};
	// 定义了一个函数 solveQuadraticEquation，用于解二次方程，并返回一个存储解的 boost::container::static_vector（double类型的二维向量）。
	auto solveQuadraticEquation(double a, double b, double c) -> boost::container::static_vector<double, 2> {
		// 计算判别式b^2 - 4ac
		const double discriminat = std::pow(b, 2) - 4 * a * c;
		// 小于0无解
		if (discriminat < 0)
			return {};

		// 等于0只有一个解 解为对称轴 -b/2
		if (discriminat == 0)
			return { -b / 2 * a };

		// 计算出两个解x1和x2并返回，
		const auto x1 = (-b - std::sqrt(discriminat)) / 2 * a;
		const auto x2 = (-b + std::sqrt(discriminat)) / 2 * a;
		// 返回的这两个解的类型为boost::container::static_vector<double, 2>
		return { x1, x2 };
	}

	void extractSphere(int resolution) {
		// 一个 Sphere 结构体被创建，表示一个球体，其中心位于原点 (0, 0, 0)，半径为 5
		const auto sphere = Sphere{glm::vec3{0, 0, 0}, 5};
		// 通过球体的中心和半径，创建了一个包围盒 BoundingBox
		const auto box = BoundingBox{sphere.center - sphere.radius, sphere.center + sphere.radius};
		// 这个函数的作用是进行 TriDexel 算法，这是一种用于生成三维体素表示的算法，通常在体素化、体素渲染等领域中应用。
		// 輸入参数1.包围盒box，2.resolution分辨率，3.回调函数RaycastCallback rcc
		// 返回一个包含三角形的向量 std::vector<Triangle>    Triangle这个模板在头文件tridexel.h中有详细说明
		const auto triangles = tridexel(box, resolution, [&](Ray ray, HitCallback hc) {
			// from https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection
			// solve quadratic equation
			// 根据射线的原点和球体的中心计算射线与球体之间的向量 L
			const auto L = ray.origin - sphere.center;
			// 定义二次方程的系数 a、b 和 c，用于计算射线与球体的交点
			const auto a = 1.0;
			const auto b = 2 * glm::dot(ray.direction, L);
			const auto c = glm::dot(L, L) - std::pow(sphere.radius, 2);
			// 求解二次方程，得到可能的交点参数 t 的值
			const auto solutions = solveQuadraticEquation(a, b, c);
			// 遍历 solutions 向量中的每个 t 值，计算交点的实际位置 point 和法线 normal
			for (const auto& t : solutions) {
				const auto point = ray.origin + (float)t * ray.direction;
				const auto normal = glm::normalize(point - sphere.center);
				// 调用 hc 函数，将交点位置和法线信息传递给它。hc 是一个 HitCallback 类型的函数对象，在这个函数中，交点的深度和法线信息会被保存下来
				hc(glm::dot(ray.origin, ray.direction) + (float)t, normal);
			}
		});
		// 将体素化后的三角面片数据保存为一个 STL 文件
		saveTriangles("sphere" + std::to_string(resolution) + ".stl", triangles);
	}


	std::vector<STLTrans::Triangle> ReadSTL(const std::string& filePath){
		Head head;
		FILE *fp;
		char fileName[128];
		strcpy(fileName, filePath.c_str());
		fp = fopen(fileName, "rb");

		if (fp != NULL)
		{
			fread(head.partName, 80, 1, fp); // 获取部件名
			fread(&head.faceNum, 4, 1, fp);	 // 获取三角面片的数目
		}

		std::vector<STLTrans::Triangle> triangles;
		// 循环读取三角面片数据
		for (int i = 0; i < head.faceNum; i++)
		{
			STLTrans::Triangle trangle;
			fread(&trangle.normal, 12, 1, fp);	// 读取法线数据
			fread(&trangle.vertex1, 12, 1, fp); // 读取顶点1的数据
			fread(&trangle.vertex2, 12, 1, fp); // 读取顶点2的数据
			fread(&trangle.vertex3, 12, 1, fp); // 读取顶点3的数据
			fread(&trangle.info, 2, 1, fp);		// 读取保留项数据，这一项一般没什么用，这里选择读取是为了移动文件指针
			Calculate::calculateCircumcircle(trangle);
			triangles.push_back(trangle);
		}

		fclose(fp);

		return triangles; // 返回包含三角形数据的向量

	}

	void WorkpieceTransTridexel(unsigned int resolution)
	{

		std::vector<STLTrans::Triangle> trianglesWork = ReadSTL("./搞个球.STL");
		

		float WorkminX = FLT_MAX, WorkminY = FLT_MAX, WorkminZ = FLT_MAX;
		float WorkmaxX = FLT_MIN, WorkmaxY = FLT_MIN, WorkmaxZ = FLT_MIN;

		// 调用CalculateBox函数来计算外包围盒
		Calculate::CalculateBox(trianglesWork, WorkminX, WorkminY, WorkminZ, WorkmaxX, WorkmaxY, WorkmaxZ);

		BoundingBox Workbox;

		if(WorkminX>0){WorkminX = WorkminX*0.9;}else{WorkminX = WorkminX*1.1;}
		if(WorkminY>0){WorkminY = WorkminY*0.9;}else{WorkminY = WorkminY*1.1;}
		if(WorkminZ>0){WorkminZ = WorkminZ*0.9;}else{WorkminZ = WorkminZ*1.1;}
		if(WorkmaxX<0){WorkmaxX = WorkmaxX*0.9;}else{WorkmaxX = WorkmaxX*1.1;}
		if(WorkmaxY<0){WorkmaxY = WorkmaxY*0.9;}else{WorkmaxY = WorkmaxY*1.1;}
		if(WorkmaxZ<0){WorkmaxZ = WorkmaxZ*0.9;}else{WorkmaxZ = WorkmaxZ*1.1;}
		if(WorkminX == 0){WorkminX -= 1; }
		if(WorkminY == 0){WorkminY -= 1; }
		if(WorkminZ == 0){WorkminZ -= 1; }
		if(WorkmaxX == 0){WorkmaxX += 1; }
		if(WorkmaxY == 0){WorkmaxY += 1; }
		if(WorkmaxZ == 0){WorkmaxZ += 1; }

		glm::vec3 WorkminPoint(WorkminX, WorkminY, WorkminZ);
		Workbox.lower = WorkminPoint;
		glm::vec3 WorkmaxPoint(WorkmaxX, WorkmaxY, WorkmaxZ);
		Workbox.upper = WorkmaxPoint;
		// 计算包围盒的每个维度的尺寸（大小），即 boxSizes，这是通过将上界减去下界得到的。
		const auto boxSizes = Workbox.upper - Workbox.lower;
		// 从 boxSizes 中找到最大的尺寸值，表示在包围盒的三个维度中，最大的尺寸值。这个最大的尺寸值用于确保在所有维度上都有足够的均匀分辨率体素。
		const auto largest = std::max({boxSizes.x, boxSizes.y, boxSizes.z});
		// 计算每个体素的大小 cellSize，即将最大尺寸值除以分辨率
		const auto cellSize = largest / resolution;

		auto res = uvec3{glm::round(boxSizes / cellSize)};

		std::cout<<"res is :\n" << res[0] <<"\n" << res[1] << "\n" <<res[2]<<"\n";

		std::vector<RayTrans> WorkrayCollectionXY;

		


		std::cout<< "AABBWork is(X) : " << WorkminX <<" " << WorkmaxX<< "\n";
		std::cout<< "AABBWork is(Y) : " << WorkminY <<" " << WorkmaxY<< "\n";
		std::cout<< "AABBWork is(Z) : " << WorkminZ <<" " << WorkmaxZ<< "\n";
		



		for (float x = WorkminX; x <= WorkmaxX; x += cellSize)
		{
			for (float y = WorkminY; y <= WorkmaxY; y += cellSize)
			{
				RayTrans ray;
				ray.origin = glm::vec3(x, y, WorkminZ);			 // Start at each point (x, y, 0)
				ray.direction = glm::vec3(0.0f, 0.0f, 1.0f); // Direction towards positive Z axis
				WorkrayCollectionXY.push_back(ray);
				// std::cout<< ray.direction.x << " "<< ray.direction.y<< " "<< ray.direction.z << "\n"
				//   << ray.origin.x <<" "<< ray.origin.y <<" "<< ray.origin.z <<"\n\n";
			}
		}

		std::vector<RayTrans> WorkrayCollectionXZ;
		for (float x = WorkminX; x <= WorkmaxX; x += cellSize)
		{
			for (float z = WorkminZ; z <= WorkmaxZ; z += cellSize)
			{
				RayTrans ray;
				ray.origin = glm::vec3(x, WorkminY, z);			 // Start at each point (x, y, 0)
				ray.direction = glm::vec3(0.0f, 1.0f, 0.0f); // Direction towards positive Z axis
				WorkrayCollectionXZ.push_back(ray);
			}
		}

		std::vector<RayTrans> WorkrayCollectionYZ;
		for (float y = WorkminY; y <= WorkmaxY; y += cellSize)
		{
			for (float z = WorkminZ; z <= WorkmaxZ; z += cellSize)
			{
				RayTrans ray;
				ray.origin = glm::vec3(WorkminX, y, z);			 // Start at each point (x, y, 0)
				ray.direction = glm::vec3(1.0f, 0.0f, 0.0f); // Direction towards positive Z axis
				WorkrayCollectionYZ.push_back(ray);
			}
		}

		Calculate::BVHNode* Workroot = Calculate::BuildBVHTree(trianglesWork, 0, trianglesWork.size() - 1);

		std::vector<glm::vec3> WorkintersectionPointsXY;
		std::vector<std::vector<glm::vec3>> WorkrightIntersectionPointsXY; // 存储满足条件的交点集合
		long num11=0;
		for (const RayTrans &ray : WorkrayCollectionXY)
		{
			
			IntersectBVHNode(Workroot, ray, trianglesWork, WorkintersectionPointsXY);
			if(WorkintersectionPointsXY.size() %2==0 && WorkintersectionPointsXY.size()!=0){		
				WorkrightIntersectionPointsXY.push_back(WorkintersectionPointsXY);
				
			}
			if(num11 ==30 || num11==31)
			{
				for (const auto& point : WorkintersectionPointsXY) {
					std::cout << num11<< "   Intersection Point: (" << point.x << ", " << point.y << ", " << point.z << ")\n";
				}
				std::cout << "ray direction:(" <<ray.direction.x<<","<<ray.direction.y << "," << ray.direction.z<<")\n";
				std::cout << "ray origin:(" <<ray.origin.x<<","<<ray.origin.y << "," << ray.origin.z<<")\n";
			}
			
			num11++;
			WorkintersectionPointsXY.clear();
		}

		// 合并所有交点到一个单独的容器
		std::vector<glm::vec3> WorkallIntersectionPointsXY;
		for (const std::vector<glm::vec3>& WorkintersectionPoints : WorkrightIntersectionPointsXY) {
			WorkallIntersectionPointsXY.insert(WorkallIntersectionPointsXY.end(), WorkintersectionPoints.begin(), WorkintersectionPoints.end());
		}

		// 使用自定义排序函数，按照 x、y、z 轴坐标升序排序
		std::sort(WorkallIntersectionPointsXY.begin(), WorkallIntersectionPointsXY.end(),
				[](const glm::vec3& a, const glm::vec3& b) {
					if (a.x != b.x) {
						return a.x < b.x;
					} else if (a.y != b.y) {
						return a.y < b.y;
					} else {
						return a.z < b.z;
					}
				});


		//插入线段容器
		std::vector<Segment> WorkSegmentXY;
		if (WorkallIntersectionPointsXY.size() % 2 == 0) {
			for (size_t i = 0; i < WorkallIntersectionPointsXY.size(); i += 2) {
				// Create a segment using the current and next point
				Segment segment(WorkallIntersectionPointsXY[i], WorkallIntersectionPointsXY[i + 1]);
				// Add the segment to the vector
				WorkSegmentXY.push_back(segment);
			}
		}




		std::vector<glm::vec3> WorkintersectionPointsYZ;
		std::vector<std::vector<glm::vec3>> WorkrightIntersectionPointsYZ; // 存储满足条件的交点集合
		for (const RayTrans &ray : WorkrayCollectionYZ)
		{
			IntersectBVHNode(Workroot, ray, trianglesWork, WorkintersectionPointsYZ);
			if(WorkintersectionPointsYZ.size() %2==0 && WorkintersectionPointsYZ.size()!=0){
				WorkrightIntersectionPointsYZ.push_back(WorkintersectionPointsYZ);
			}
			WorkintersectionPointsYZ.clear();			
		}

		std::vector<glm::vec3> WorkallIntersectionPointsYZ;

		for (const std::vector<glm::vec3>& WorkintersectionPoints : WorkrightIntersectionPointsYZ) {
			WorkallIntersectionPointsYZ.insert(WorkallIntersectionPointsYZ.end(), WorkintersectionPoints.begin(), WorkintersectionPoints.end());
		}

		// 使用自定义排序函数，按照 y、z、x 轴坐标升序排序
		std::sort(WorkallIntersectionPointsYZ.begin(), WorkallIntersectionPointsYZ.end(),
				[](const glm::vec3& a, const glm::vec3& b) {
					if (a.y != b.y) {
						return a.y < b.y;
					} else if (a.z != b.z) {
						return a.z < b.z;
					} else {
						return a.x < b.x;
					}
				});

		//插入线段容器
		std::vector<Segment> WorkSegmentYZ;
		if (WorkallIntersectionPointsYZ.size() % 2 == 0) {
			for (size_t i = 0; i < WorkallIntersectionPointsYZ.size(); i += 2) {
				// Create a segment using the current and next point
				Segment segment(WorkallIntersectionPointsYZ[i], WorkallIntersectionPointsYZ[i + 1]);
				// Add the segment to the vector
				WorkSegmentYZ.push_back(segment);
			}
		}

		std::vector<glm::vec3> WorkintersectionPointsXZ;
		std::vector<std::vector<glm::vec3>> WorkrightIntersectionPointsXZ; // 存储满足条件的交点集合
		for (const RayTrans &ray : WorkrayCollectionXZ)
		{
			IntersectBVHNode(Workroot, ray, trianglesWork, WorkintersectionPointsXZ);
			if(WorkintersectionPointsXZ.size() %2 == 0 && WorkintersectionPointsXZ.size()!=0){

				WorkrightIntersectionPointsXZ.push_back(WorkintersectionPointsXZ);
				
			}
			WorkintersectionPointsXZ.clear();
		}
		std::vector<glm::vec3> WorkallIntersectionPointsXZ;

		for (const std::vector<glm::vec3>& WorkintersectionPoints : WorkrightIntersectionPointsXZ) {
			WorkallIntersectionPointsXZ.insert(WorkallIntersectionPointsXZ.end(), WorkintersectionPoints.begin(), WorkintersectionPoints.end());
		}

		// 使用自定义排序函数，按照 x、y、z 轴坐标升序排序
		std::sort(WorkallIntersectionPointsXZ.begin(), WorkallIntersectionPointsXZ.end(),
				[](const glm::vec3& a, const glm::vec3& b) {
					if (a.x != b.x) {
						return a.x < b.x;
					} else if (a.z != b.z) {
						return a.z < b.z;
					} else {
						return a.y < b.y;
					}
				});

		//插入线段容器
		std::vector<Segment> WorkSegmentXZ;
		if (WorkallIntersectionPointsXZ.size() % 2 == 0) {
			for (size_t i = 0; i < WorkallIntersectionPointsXZ.size(); i += 2) {
				// Create a segment using the current and next point
				Segment segment(WorkallIntersectionPointsXZ[i], WorkallIntersectionPointsXZ[i + 1]);
				// Add the segment to the vector
				WorkSegmentXZ.push_back(segment);
			}
		}


		

		// 打开用于写入的输出文件(线段容器)
		std::ofstream outFileSEXY("SEintersection_pointsXY.txt");
		if (!outFileSEXY.is_open()) {
			std::cerr << "无法打开输出文件" << std::endl;
			return;
		}
		// 将线段XY交点写入输出文件
		for (const Segment &segment : WorkSegmentXY) {
			outFileSEXY << segment.point1.x << " " << segment.point1.y << " " << segment.point1.z << "\n"<<
			               segment.point2.x << " " << segment.point2.y << " " << segment.point2.z << std::endl;
		}
		// // 关闭输出文件
		outFileSEXY.close();



		// 打开用于写入的输出文件(线段容器)
		std::ofstream outFileSEYZ("SEintersection_pointsYZ.txt");
		if (!outFileSEYZ.is_open()) {
			std::cerr << "无法打开输出文件" << std::endl;
			return;
		}
		// 将线段YZ交点写入输出文件
		for (const Segment &segment : WorkSegmentYZ) {
			outFileSEYZ << segment.point1.x << " " << segment.point1.y << " " << segment.point1.z << "\n"<<
			               segment.point2.x << " " << segment.point2.y << " " << segment.point2.z << std::endl;
		}
		// // 关闭输出文件
		outFileSEYZ.close();




		// 打开用于写入的输出文件(线段容器)
		std::ofstream outFileSEXZ("SEintersection_pointsXZ.txt");
		if (!outFileSEXZ.is_open()) {
			std::cerr << "无法打开输出文件" << std::endl;
			return;
		}
		// 将线段XZ交点写入输出文件
		for (const Segment &segment : WorkSegmentXZ) {
			outFileSEXZ << segment.point1.x << " " << segment.point1.y << " " << segment.point1.z << "\n"<<
			               segment.point2.x << " " << segment.point2.y << " " << segment.point2.z << std::endl;
		}
		// // 关闭输出文件
		outFileSEXZ.close();

		
		

		Calculate::DeleteBVHTree(Workroot);

		std::vector<STLTrans::Triangle> trianglesKnife = ReadSTL("./搞个屁哦.STL");

		float KnifeminX = FLT_MAX, KnifeminY = FLT_MAX, KnifeminZ = FLT_MAX;
		float KnifemaxX = FLT_MIN, KnifemaxY = FLT_MIN, KnifemaxZ = FLT_MIN;

		// 调用CalculateBox函数来计算外包围盒
		Calculate::CalculateBox(trianglesKnife, KnifeminX, KnifeminY, KnifeminZ, KnifemaxX, KnifemaxY, KnifemaxZ);

		BoundingBox Knifebox;

		if(KnifeminX>0){KnifeminX = KnifeminX*0.9;}else{KnifeminX = KnifeminX*1.1;}
		if(KnifeminY>0){KnifeminY = KnifeminY*0.9;}else{KnifeminY = KnifeminY*1.1;}
		if(KnifeminZ>0){KnifeminZ = KnifeminZ*0.9;}else{KnifeminZ = KnifeminZ*1.1;}
		if(KnifemaxX<0){KnifemaxX = KnifemaxX*0.9;}else{KnifemaxX = KnifemaxX*1.1;}
		if(KnifemaxY<0){KnifemaxY = KnifemaxY*0.9;}else{KnifemaxY = KnifemaxY*1.1;}
		if(KnifemaxZ<0){KnifemaxZ = KnifemaxZ*0.9;}else{KnifemaxZ = KnifemaxZ*1.1;}
		if(KnifeminX == 0){KnifeminX -= 1; }
		if(KnifeminY == 0){KnifeminY -= 1; }
		if(KnifeminZ == 0){KnifeminZ -= 1; }
		if(KnifemaxX == 0){KnifemaxX += 1; }
		if(KnifemaxY == 0){KnifemaxY += 1; }
		if(KnifemaxZ == 0){KnifemaxZ += 1; }



		glm::vec3 KnifeminPoint(KnifeminX, KnifeminY, KnifeminZ);
		Knifebox.lower = KnifeminPoint;
		glm::vec3 KnifemaxPoint(KnifemaxX, KnifemaxY, KnifemaxZ);
		Knifebox.upper = KnifemaxPoint;

		Calculate::BVHNode* Kniferoot = Calculate::BuildBVHTree(trianglesKnife, 0, trianglesKnife.size() - 1);

		// 遍历WorkSegmentXY容器，将线段按是否落入矩形区域内分为俩个部分
		std::vector<Segment> segmentsInRectangleXY;
		std::vector<Segment> segmentsUnInRectangleXY;
		std::vector<Segment> EDsegmentsInRectangleXY;
		// std::cout << KnifeminX <<"   "<<KnifeminY <<"   "<<KnifeminZ <<"\n"
		//           << KnifemaxX <<"   "<<KnifemaxY <<"   "<<KnifemaxZ <<"\n";


		for (Segment &segment : WorkSegmentXY)
		{
			glm::vec3 p1 = segment.point1;
			glm::vec3 p2 = segment.point2;
			// std::cout<< segment.point1.x <<" " <<segment.point1.y <<" " <<segment.point1.z <<"    "<<
			// 	segment.point2.x <<" " <<segment.point2.y<<" " <<segment.point2.z << "\n";
			
			// 判断线段是否落入矩形区域
			if ((p1.x >= KnifeminX && p1.x <= KnifemaxX && p1.y >= KnifeminY && p1.y <= KnifemaxY) &&
				(p2.x >= KnifeminX && p2.x <= KnifemaxX && p2.y >= KnifeminY && p2.y <= KnifemaxY))
			{
				segmentsInRectangleXY.push_back(segment);
				// std::cout<< segment.point1.x <<" " <<segment.point1.y <<" " <<segment.point1.z <<"    "<<
				// segment.point2.x <<" " <<segment.point2.y<<" " <<segment.point2.z << "\n";
				RayTrans ray;
				ray.origin = glm::vec3(p1.x, p1.y, KnifeminZ);	
				ray.direction = glm::vec3(0.0f, 0.0f, 1.0f);
				std::vector<glm::vec3> KnifeintersectionPointsXY;
				std::vector<std::vector<glm::vec3>> KniferightIntersectionPointsXY; // 存储满足条件的交点集合
				IntersectBVHNode(Kniferoot, ray, trianglesKnife, KnifeintersectionPointsXY);
				if(KnifeintersectionPointsXY.size() %2==0 && KnifeintersectionPointsXY.size()!=0){		
					std::sort(KnifeintersectionPointsXY.begin(), KnifeintersectionPointsXY.end(),
						[](const glm::vec3& a, const glm::vec3& b) {
							if (a.x != b.x) {
								return a.x < b.x;
							} else if (a.y != b.y) {
								return a.y < b.y;
							} else {
								return a.z < b.z;
							}
						});
					std::vector<Segment> KnifeSegmentXY;
					for (size_t i = 0; i < KnifeintersectionPointsXY.size(); i += 2) {
						// Create a segment using the current and next point
						Segment segment(KnifeintersectionPointsXY[i], KnifeintersectionPointsXY[i + 1]);
						// Add the segment to the vector
						KnifeSegmentXY.push_back(segment);
					}
					for(Segment &segmentXY : KnifeSegmentXY)
					{
						// std::cout <<"刀具线段segmentXY\n"
						// 		  << segmentXY.point1.x << " " << segmentXY.point1.y << " " << segmentXY.point1.z << "\n"
						// 		  << segmentXY.point2.x << " " << segmentXY.point2.y << " " << segmentXY.point2.z << "\n"
						// 		  <<"原来毛坯线段\n"
						//           << segment.point1.x << " " << segment.point1.y << " " << segment.point1.z << "\n"
						// 		  << segment.point2.x << " " << segment.point2.y << " " << segment.point2.z << "\n";
						if( p2.z < segmentXY.point1.z||p1.z > segmentXY.point2.z )
						{
							// std::cout << "1";
							EDsegmentsInRectangleXY.push_back(segment);
						}
						else if(p1.z<segmentXY.point1.z && segmentXY.point1.z<p2.z && p2.z<segmentXY.point2.z)
						{
							// std::cout << "2";
							segment.point2.z = segmentXY.point1.z;
							EDsegmentsInRectangleXY.push_back(segment);
						}
						else if(segmentXY.point1.z<p1.z && p1.z<p2.z && p2.z<segmentXY.point2.z)
						{
													
						}
						else if(p1.z<segmentXY.point1.z && segmentXY.point1.z < segmentXY.point2.z && segmentXY.point2.z<p2.z)
						{
							auto a = segment.point2.z;
							segment.point2.z = segmentXY.point1.z;
							EDsegmentsInRectangleXY.push_back(segment);
							segment.point1.z = segmentXY.point2.z;
							segment.point2.z = a;
							EDsegmentsInRectangleXY.push_back(segment);	
						}
						else if(segmentXY.point1.z < p1.z && p1.z<segmentXY.point2.z && segmentXY.point2.z<p2.z)
						{
							segment.point1.z = segmentXY.point2.z;
							EDsegmentsInRectangleXY.push_back(segment);	
						}
					}
				}
				else if(KnifeintersectionPointsXY.size() == 0)
				{
					EDsegmentsInRectangleXY.push_back(segment);	
				}
				KnifeintersectionPointsXY.clear();
			}
			else
			{
				segmentsUnInRectangleXY.push_back(segment);
			}
		}
		// 遍历K你份额SegmentYZ容器，将线段按是否落入矩形区域内分为俩个部分
		std::vector<Segment> segmentsInRectangleYZ;
		std::vector<Segment> segmentsUnInRectangleYZ;
		std::vector<Segment> EDsegmentsInRectangleYZ;
		for (Segment &segment : WorkSegmentYZ)
		{
			glm::vec3 p1 = segment.point1;
			glm::vec3 p2 = segment.point2;

			// 判断线段是否落入矩形区域
			if ((p1.y >= KnifeminY && p1.y <= KnifemaxY && p1.z >= KnifeminZ && p1.z <= KnifemaxZ) &&
				(p2.y >= KnifeminY && p2.y <= KnifemaxY && p2.z >= KnifeminZ && p2.z <= KnifemaxZ))
			{
				segmentsInRectangleYZ.push_back(segment);
				RayTrans ray;
				ray.origin = glm::vec3(KnifeminX, p1.y, p1.z);	
				ray.direction = glm::vec3(1.0f, 0.0f, 0.0f);
				std::vector<glm::vec3> KnifeintersectionPointsYZ;
				std::vector<std::vector<glm::vec3>> KniferightIntersectionPointsYZ; // 存储满足条件的交点集合
				IntersectBVHNode(Kniferoot, ray, trianglesKnife, KnifeintersectionPointsYZ);
				if(KnifeintersectionPointsYZ.size() %2==0 && KnifeintersectionPointsYZ.size()!=0){		
					std::sort(KnifeintersectionPointsYZ.begin(), KnifeintersectionPointsYZ.end(),
						[](const glm::vec3& a, const glm::vec3& b) {
							if (a.y != b.y) {
								return a.y < b.y;
							} else if (a.z != b.z) {
								return a.z < b.z;
							} else {
								return a.x < b.x;
							}
						});
					std::vector<Segment> KnifeSegmentYZ;
					for (size_t i = 0; i < KnifeintersectionPointsYZ.size(); i += 2) {
						// Create a segment using the current and next point
						Segment segment(KnifeintersectionPointsYZ[i], KnifeintersectionPointsYZ[i + 1]);
						// Add the segment to the vector
						KnifeSegmentYZ.push_back(segment);
					}
					for(Segment &segmentYZ : KnifeSegmentYZ)
					{
						// std::cout <<"刀具线段segmentYZ\n"
						// 		  << segmentYZ.point1.x << " " << segmentYZ.point1.y << " " << segmentYZ.point1.z << "\n"
						// 		  << segmentYZ.point2.x << " " << segmentYZ.point2.y << " " << segmentYZ.point2.z << "\n"
						// 		  <<"原来毛坯线段\n"
						//           << segment.point1.x << " " << segment.point1.y << " " << segment.point1.z << "\n"
						// 		  << segment.point2.x << " " << segment.point2.y << " " << segment.point2.z << "\n";
						if( p2.x < segmentYZ.point1.x||p1.x > segmentYZ.point2.x )
						{
							// std::cout << "1";
							EDsegmentsInRectangleYZ.push_back(segment);
						}
						else if(p1.x<segmentYZ.point1.x && segmentYZ.point1.x<p2.x && p2.x<segmentYZ.point2.x)
						{
							// std::cout << "2";
							segment.point2.x = segmentYZ.point1.x;
							EDsegmentsInRectangleYZ.push_back(segment);
						}
						else if(segmentYZ.point1.x<p1.x && p1.x<p2.x && p2.x<segmentYZ.point2.x)
						{
													
						}
						else if(p1.x<segmentYZ.point1.x && segmentYZ.point1.x < segmentYZ.point2.x && segmentYZ.point2.x<p2.x)
						{
							auto a = segment.point2.x;
							segment.point2.x = segmentYZ.point1.x;
							EDsegmentsInRectangleYZ.push_back(segment);
							segment.point1.x = segmentYZ.point2.x;
							segment.point2.x = a;
							EDsegmentsInRectangleYZ.push_back(segment);	
						}
						else if(segmentYZ.point1.x < p1.x && p1.x<segmentYZ.point2.x && segmentYZ.point2.x<p2.x)
						{
							segment.point1.x = segmentYZ.point2.x;
							EDsegmentsInRectangleYZ.push_back(segment);	
						}
					}
				}
				else if(KnifeintersectionPointsYZ.size() == 0)
				{
					EDsegmentsInRectangleYZ.push_back(segment);	
				}
				KnifeintersectionPointsYZ.clear();
			}
			else
			{
				segmentsUnInRectangleYZ.push_back(segment);
			}
		}
		// 遍历WorkSegmentXZ容器，将线段按是否落入矩形区域内分为俩个部分
		std::vector<Segment> segmentsInRectangleXZ;
		std::vector<Segment> segmentsUnInRectangleXZ;
		std::vector<Segment> EDsegmentsInRectangleXZ;
		for (Segment &segment : WorkSegmentXZ)
		{
			glm::vec3 p1 = segment.point1;
			glm::vec3 p2 = segment.point2;

			// 判断线段是否落入矩形区域
			if ((p1.x >= KnifeminX && p1.x <= KnifemaxX && p1.z >= KnifeminZ && p1.z <= KnifemaxZ) &&
				(p2.x >= KnifeminX && p2.x <= KnifemaxX && p2.z >= KnifeminZ && p2.z <= KnifemaxZ))
			{
				segmentsInRectangleXZ.push_back(segment);
				RayTrans ray;
				ray.origin = glm::vec3(p1.x, KnifeminY, p1.z);	
				ray.direction = glm::vec3(0.0f, 1.0f, 0.0f);
				std::vector<glm::vec3> KnifeintersectionPointsXZ;
				std::vector<std::vector<glm::vec3>> KniferightIntersectionPointsXZ; // 存储满足条件的交点集合
				IntersectBVHNode(Kniferoot, ray, trianglesKnife, KnifeintersectionPointsXZ);
				if(KnifeintersectionPointsXZ.size() %2==0 && KnifeintersectionPointsXZ.size()!=0){		
					std::sort(KnifeintersectionPointsXZ.begin(), KnifeintersectionPointsXZ.end(),
						[](const glm::vec3& a, const glm::vec3& b) {
							if (a.x != b.x) {
								return a.x < b.x;
							} else if (a.z != b.z) {
								return a.z < b.z;
							} else {
								return a.y < b.y;
							}
						});
					std::vector<Segment> KnifeSegmentXZ;
					for (size_t i = 0; i < KnifeintersectionPointsXZ.size(); i += 2) {
						// Create a segment using the current and next point
						Segment segment(KnifeintersectionPointsXZ[i], KnifeintersectionPointsXZ[i + 1]);
						// Add the segment to the vector
						KnifeSegmentXZ.push_back(segment);
					}
					for(Segment &segmentXZ : KnifeSegmentXZ)
					{
						// std::cout <<"刀具线段segmentXZ\n"
						// 		  << segmentXZ.point1.x << " " << segmentXZ.point1.y << " " << segmentXZ.point1.z << "\n"
						// 		  << segmentXZ.point2.x << " " << segmentXZ.point2.y << " " << segmentXZ.point2.z << "\n"
						// 		  <<"原来毛坯线段\n"
						//           << segment.point1.x << " " << segment.point1.y << " " << segment.point1.z << "\n"
						// 		  << segment.point2.x << " " << segment.point2.y << " " << segment.point2.z << "\n";
						if( p2.y < segmentXZ.point1.y||p1.y > segmentXZ.point2.y )
						{
							// std::cout << "1";
							EDsegmentsInRectangleXZ.push_back(segment);
						}
						else if(p1.y<segmentXZ.point1.y && segmentXZ.point1.y<p2.y && p2.y<segmentXZ.point2.y)
						{
							// std::cout << "2";
							segment.point2.y = segmentXZ.point1.y;
							EDsegmentsInRectangleXZ.push_back(segment);
						}
						else if(segmentXZ.point1.y<p1.y && p1.y<p2.y && p2.y<segmentXZ.point2.y)
						{
													
						}
						else if(p1.y<segmentXZ.point1.y && segmentXZ.point1.y < segmentXZ.point2.y && segmentXZ.point2.y<p2.y)
						{
							auto a = segment.point2.y;
							segment.point2.y = segmentXZ.point1.y;
							EDsegmentsInRectangleXZ.push_back(segment);
							segment.point1.y = segmentXZ.point2.y;
							segment.point2.y = a;
							EDsegmentsInRectangleXZ.push_back(segment);	
						}
						else if(segmentXZ.point1.y < p1.y && p1.y<segmentXZ.point2.y && segmentXZ.point2.y<p2.y)
						{
							segment.point1.y = segmentXZ.point2.y;
							EDsegmentsInRectangleXZ.push_back(segment);	
						}
					}
				}
				else if(KnifeintersectionPointsXZ.size() == 0)
				{
					EDsegmentsInRectangleXZ.push_back(segment);	
				}
				KnifeintersectionPointsXZ.clear();
			}
			else
			{
				segmentsUnInRectangleXZ.push_back(segment);
			}
		}



		// 打开用于写入的输出文件(线段容器)
		std::ofstream outFileTE("TEintersection_pointsXZ.txt");
		if (!outFileTE.is_open()) {
			std::cerr << "无法打开输出文件" << std::endl;
			return;
		}
		// 将线段XY交点写入输出文件
		for (const Segment &segment : EDsegmentsInRectangleXY) {
			outFileTE << segment.point1.x << " " << segment.point1.y << " " << segment.point1.z << "\n"<<
			               segment.point2.x << " " << segment.point2.y << " " << segment.point2.z << std::endl;
		}
		for (const Segment &segment : segmentsUnInRectangleXY) {
			outFileTE << segment.point1.x << " " << segment.point1.y << " " << segment.point1.z << "\n"<<
			               segment.point2.x << " " << segment.point2.y << " " << segment.point2.z << std::endl;
		}
		for (const Segment &segment : EDsegmentsInRectangleYZ) {
			outFileTE << segment.point1.x << " " << segment.point1.y << " " << segment.point1.z << "\n"<<
			               segment.point2.x << " " << segment.point2.y << " " << segment.point2.z << std::endl;
		}
		for (const Segment &segment : segmentsUnInRectangleYZ) {
			outFileTE << segment.point1.x << " " << segment.point1.y << " " << segment.point1.z << "\n"<<
			               segment.point2.x << " " << segment.point2.y << " " << segment.point2.z << std::endl;
		}
		for (const Segment &segment : EDsegmentsInRectangleXZ) {
			outFileTE << segment.point1.x << " " << segment.point1.y << " " << segment.point1.z << "\n"<<
			               segment.point2.x << " " << segment.point2.y << " " << segment.point2.z << std::endl;
		}
		for (const Segment &segment : segmentsUnInRectangleXZ) {
			outFileTE << segment.point1.x << " " << segment.point1.y << " " << segment.point1.z << "\n"<<
			               segment.point2.x << " " << segment.point2.y << " " << segment.point2.z << std::endl;
		}
		// 关闭输出文件
		outFileTE.close();



	}

	
}


// TEST(sphere, extract5) {
// 	extractSphere(5);
// }

TEST(sphere, 2) {
	WorkpieceTransTridexel(100);
}

// TEST(sphere, 2) {
// 	std::cout << "1" <<std::endl;
// }

// TEST(sphere, 3) {
// 	KnifeTransTridexel();
// }

// TEST(sphere, extract10) {
// 	extractSphere(10);
// }

// TEST(sphere, extract100) {
// 	extractSphere(100);
// }

// TEST(sphere, extract200) {
// 	extractSphere(200);
// }
