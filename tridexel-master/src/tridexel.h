#pragma once

#include <glm/vec3.hpp>
#include <glm/geometric.hpp>

#include <vector>
#include <array>
#include <functional>


// Ray 结构体表示一个射线，具有两个成员变量
struct Ray {
	// origin：表示射线的起始点，是一个 glm::vec3 类型的向量，存储了射线的起始坐标。
	glm::vec3 origin;
	// direction：表示射线的方向，也是一个 glm::vec3 类型的向量，存储了射线的方向向量。
	glm::vec3 direction;
};

// BoundingBox 结构体表示一个包围盒，用于限定一个空间区域，具有两个成员变量
struct BoundingBox {
	// lower：表示包围盒的下界，是一个 glm::vec3 类型的向量，存储了包围盒的最小坐标点。
	glm::vec3 lower;
	// upper：表示包围盒的上界，是一个 glm::vec3 类型的向量，存储了包围盒的最大坐标点。
	glm::vec3 upper;
};

// 定义了一个名为 Triangle 的结构体，它继承自 std::array<glm::vec3, 3>，表示一个由三个 glm::vec3 类型的点构成的三角形。
// 这种方式通过继承 std::array 来定义三角形，使得 Triangle 本质上是一个固定大小为 3 的数组，每个元素都是 glm::vec3 类型的点坐标。
// 这种定义方式使得可以通过数组索引来访问三角形的三个顶点。
struct Triangle : std::array<glm::vec3, 3> {};


// 使用 C++ 中的类型别名（Type Alias）来定义了两个函数类型：HitCallback 和 RaycastCallback
// HitCallback 是一个函数类型，它接受两个参数，一个是 float 类型的参数，表示击中点的深度（或距离），
// 另一个是 glm::vec3 类型的参数，表示击中点的法线向量。这个函数类型表示一个在射线与物体相交时调用的回调函数，用于处理相交点的信息
using HitCallback = std::function<void(float, glm::vec3)>;
// RaycastCallback 是一个函数类型，它接受两个参数，一个是 Ray 类型的参数，表示光线（射线）的信息，另一个是 HitCallback 类型的参数，
// 表示一个回调函数。这个函数类型表示一个射线投射的过程，其中光线与物体相交时，会调用传递进来的回调函数来处理相交点的信息。
using RaycastCallback = std::function<void(Ray, HitCallback)>;


// // 这个函数的作用是进行 TriDexel 算法，这是一种用于生成三维体素表示的算法，通常在体素化、体素渲染等领域中应用。
// 輸入参数1.包围盒box，2.resolution分辨率，3.回调函数RaycastCallback rcc
// 返回一个包含三角形的向量 std::vector<Triangle>    Triangle这个模板在头文件tridexel.h中有详细说明
auto tridexel(BoundingBox box, unsigned int resolution, RaycastCallback rcc) -> std::vector<Triangle>;





/*
		using 是 C++ 中的一个关键字，用于定义类型别名（Type Alias）。它允许你创建一个新的名称（别名）来引用现有的类型，使得代码更加可读和灵活。
		类型别名通过 using 关键字可以在不创建新类型的情况下为已有类型赋予新名称。

		使用 using 关键字可以有以下几个目的：

		增加可读性： 可以用更直观的名称来表示复杂的类型，使代码更加易于理解和维护。

		简化代码： 可以缩短较长的类型名，减少重复的代码。

		泛型编程： 可以用别名来表示模板类型参数，使得模板代码更具通用性。

		封装库： 可以在库的接口中使用别名，从而使库的用户更容易理解和使用。
*/
