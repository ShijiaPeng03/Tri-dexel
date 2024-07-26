// 包含 GTest 框架头文件的指令，用于引入测试框架的定义和功能。
#include <gtest/gtest.h>
#include "../src/tridexel.h"
#include "../src/IO.h"

int main(int argc, char** argv) {
	// 初始化了 GTest 框架，并告诉框架从命令行参数中获取测试相关的信息
	::testing::InitGoogleTest(&argc, argv);
	// 运行了所有的测试用例，并返回测试的结果。它会执行测试用例中的各个测试，并输出测试的结果信息。
	return RUN_ALL_TESTS();
}
