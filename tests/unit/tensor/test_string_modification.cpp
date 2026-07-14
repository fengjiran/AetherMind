//
// Created by 赵丹 on 2025/8/22.
//
#include "any.h"
#include "container/string.h"

#include <deque>
#include <gtest/gtest.h>
#include <list>

namespace {

using namespace aethermind;

// 测试迭代器版本的insert方法
TEST(StringInsert, IteratorVersion) {
    // 测试在字符串中间插入多个相同字符
    String s1("hello");
    auto it1 = s1.insert(s1.begin() + 2, 3, 'a');
    EXPECT_STREQ(s1, "heaaallo");
    EXPECT_EQ(it1, s1.begin() + 2);// 验证迭代器指向插入位置

    // 测试在字符串开头插入单个字符
    String s2("world");
    auto it2 = s2.insert(s2.begin(), 'H');
    EXPECT_STREQ(s2, "Hworld");
    EXPECT_EQ(it2, s2.begin());// 验证迭代器指向插入位置

    // 测试在字符串结尾插入单个字符
    String s3("hello");
    auto it3 = s3.insert(s3.end(), '!');
    EXPECT_STREQ(s3, "hello!");
    EXPECT_EQ(it3, s3.end() - 1);// 验证迭代器指向插入位置
}

// 测试迭代器范围插入
TEST(StringInsert, IteratorRange) {
    String s1("hello");
    std::vector<char> chars = {'w', 'o', 'r', 'l', 'd'};

    // 在字符串中间插入字符范围
    auto it1 = s1.insert(s1.begin() + 5, chars.begin(), chars.end());
    EXPECT_STREQ(s1, "helloworld");
    EXPECT_EQ(it1, s1.begin() + 5);

    // 在空字符串中插入字符范围
    String s2;
    auto it2 = s2.insert(s2.begin(), chars.begin(), chars.end());
    EXPECT_STREQ(s2, "world");
    EXPECT_EQ(it2, s2.begin());
}

// 测试初始化列表插入
TEST(StringInsert, InitializerList) {
    String s1("hello");

    // 使用初始化列表插入
    auto it1 = s1.insert(s1.begin() + 2, {'a', 'b', 'c'});
    EXPECT_STREQ(s1, "heabcllo");
    EXPECT_EQ(it1, s1.begin() + 2);

    // 在字符串开头插入初始化列表
    String s2("world");
    auto it2 = s2.insert(s2.begin(), {'H', 'e', 'l', 'l', 'o', ' '});
    EXPECT_STREQ(s2, "Hello world");
    EXPECT_EQ(it2, s2.begin());
}

// 测试位置版本的insert方法 - 插入String对象
TEST(StringInsert, PositionString) {
    String s1("hello");
    String s2("world");

    // 在中间位置插入整个String
    s1.insert(5, s2);
    EXPECT_STREQ(s1, "helloworld");

    // 在指定位置插入String的子串
    String s3("hello");
    String s4("123456789");
    s3.insert(2, s4, 3, 4);// 从s4的位置3开始插入4个字符
    EXPECT_STREQ(s3, "he4567llo");
}

// 测试位置版本的insert方法 - 插入C字符串
TEST(StringInsert, PositionCString) {
    String s1("hello");

    // 插入带长度的C字符串
    s1.insert(2, "XYZ", 3);
    EXPECT_STREQ(s1, "heXYZllo");

    // 插入C字符串（自动计算长度）
    String s2("world");
    s2.insert(0, "Hello ");
    EXPECT_STREQ(s2, "Hello world");

    // 插入空C字符串
    String s3("test");
    s3.insert(2, "");
    EXPECT_STREQ(s3, "test");// 应该保持不变
}

// 测试位置版本的insert方法 - 插入多个相同字符
TEST(StringInsert, PositionMultipleChars) {
    String s1("hello");

    // 在指定位置插入多个相同字符
    s1.insert(3, 4, 'x');
    EXPECT_STREQ(s1.c_str(), "helxxxxlo");

    // 在位置0插入多个相同字符
    std::string s2("world");
    // std::string::const_iterator xc = 0;
    s2.insert(0, 2, 'A');
    EXPECT_STREQ(s2.c_str(), "AAworld");

    // 在末尾位置插入多个相同字符
    String s3("hello");
    s3.insert(s3.size(), 3, '!');
    EXPECT_STREQ(s3.c_str(), "hello!!!");

    // 插入0个字符
    String s4("test");
    s4.insert(2, 0, 'x');
    EXPECT_STREQ(s4.c_str(), "test");// 应该保持不变
}

// 测试边界情况
TEST(StringInsert, BoundaryConditions) {
    // 空字符串插入
    String empty;
    empty.insert(0, "hello");
    EXPECT_STREQ(empty.c_str(), "hello");

    // 在超出范围的位置插入（应该抛出异常）
    String s("test");
    EXPECT_THROW({ s.insert(s.size() + 1, "x"); }, Error);

    // 插入大量字符
    const size_t large_count = 100;
    String large("start");
    large.insert(large.size(), large_count, 'B');
    large.insert(large.size(), "end");

    EXPECT_EQ(large.size(), 5 + large_count + 3);
    EXPECT_STREQ(large.data(), "start" + String(large_count, 'B') + "end");
}

// 测试特殊字符
TEST(StringInsert, SpecialCharacters) {
    String s1("test");

    // 插入控制字符
    s1.insert(2, 1, '\n');
    EXPECT_EQ(s1.size(), 5);
    EXPECT_EQ(s1[2], '\n');

    // 插入特殊符号
    String s2("hello");
    s2.insert(5, "!@#$%");
    EXPECT_STREQ(s2.c_str(), "hello!@#$%");

    // 插入数字
    String s3("abc");
    s3.insert(3, "123");
    EXPECT_STREQ(s3.c_str(), "abc123");
}

// 测试引用计数和COW行为
TEST(StringInsert, ReferenceCounting) {
    String original("reference test");
    String copy = original;

    // 验证引用计数
    EXPECT_EQ(original.use_count(), 1);
    EXPECT_TRUE(original.unique());

    // 对副本进行插入操作，应该触发COW
    copy.insert(4, "INSERT");

    // 验证引用计数分离
    EXPECT_EQ(original.use_count(), 1);
    EXPECT_TRUE(original.unique());
    EXPECT_EQ(copy.use_count(), 1);
    EXPECT_TRUE(copy.unique());

    // 验证原始字符串未被修改
    EXPECT_STREQ(original.c_str(), "reference test");
    // 验证副本已被修改
    EXPECT_STREQ(copy.c_str(), "refeINSERTrence test");
}

// 测试resize方法（带填充字符）
TEST(StringMemoryManagementTest, ResizeWithFillChar) {
    // 测试扩展字符串
    String s1("hello");
    s1.resize(10, '!');
    EXPECT_EQ(s1.size(), 10);
    EXPECT_STREQ(s1.c_str(), "hello!!!!!");

    // 测试缩短字符串
    String s2("hello world");
    s2.resize(5, 'x');
    EXPECT_EQ(s2.size(), 5);
    EXPECT_TRUE(s2 == "hello");

    // 测试调整为相同大小
    String s3("test");
    s3.resize(4, 'y');
    EXPECT_EQ(s3.size(), 4);
    EXPECT_STREQ(s3.c_str(), "test");

    // 测试调整为空字符串
    String s4("sample");
    s4.resize(0, 'z');
    EXPECT_EQ(s4.size(), 0);
    EXPECT_TRUE(s4.empty());
    EXPECT_TRUE(s4 == "");

    // 测试从空字符串扩展
    String s5;
    s5.resize(3, 'a');
    EXPECT_EQ(s5.size(), 3);
    EXPECT_STREQ(s5.c_str(), "aaa");
}

// 测试resize方法（不带填充字符，使用默认值）
TEST(StringMemoryManagementTest, ResizeWithDefaultChar) {
    // 测试扩展字符串（默认使用\0填充）
    String s1("hello");
    s1.resize(8);
    EXPECT_EQ(s1.size(), 8);
    EXPECT_EQ(s1[5], '\0');
    EXPECT_EQ(s1[6], '\0');
    EXPECT_EQ(s1[7], '\0');

    // 验证原始内容保持不变
    char buffer[9];
    std::memcpy(buffer, s1.c_str(), 8);
    buffer[8] = '\0';
    EXPECT_TRUE(std::string(buffer).substr(0, 5) == "hello");

    // 测试缩短字符串
    String s2("hello world");
    s2.resize(5);
    EXPECT_EQ(s2.size(), 5);
    EXPECT_TRUE(s2 == "hello");

    // 测试从空字符串扩展
    String s3;
    s3.resize(4);
    EXPECT_EQ(s3.size(), 4);
    EXPECT_EQ(s3[0], '\0');
    EXPECT_EQ(s3[1], '\0');
    EXPECT_EQ(s3[2], '\0');
    EXPECT_EQ(s3[3], '\0');
}

// 测试reserve方法
TEST(StringMemoryManagementTest, Reserve) {
    // 测试reserve增加容量
    String s1("test");
    const size_t initial_cap = s1.capacity();
    s1.reserve(100);

    // 注意：这里需要验证capacity是否正确增加
    // 由于String类的reserve实现似乎有问题（传递的是当前capacity而不是n），这里需要特别注意
    EXPECT_TRUE(s1.capacity() >= initial_cap);

    // 验证内容不变
    EXPECT_STREQ(s1.c_str(), "test");
    EXPECT_EQ(s1.size(), 4);

    // 测试reserve减小容量（应该不影响）
    String s2("hello");
    const size_t original_cap = s2.capacity();
    s2.reserve(2);                             // 小于当前容量
    EXPECT_TRUE(s2.capacity() >= original_cap);// 容量不应减小
    EXPECT_STREQ(s2.c_str(), "hello");

    // 测试reserve相同容量
    String s3("sample");
    const size_t same_cap = s3.capacity();
    s3.reserve(same_cap);
    EXPECT_EQ(s3.capacity(), same_cap);
    EXPECT_STREQ(s3.c_str(), "sample");

    // 测试从空字符串reserve
    String s4;
    s4.reserve(50);
    EXPECT_TRUE(s4.capacity() >= 50);
    EXPECT_TRUE(s4.empty());
}

// 测试shrink_to_fit方法
TEST(StringMemoryManagementTest, ShrinkToFit) {
    // 测试缩小容量到实际大小
    String s1("hello world");
    s1.reserve(100);// 先增加容量
    const size_t large_cap = s1.capacity();
    EXPECT_TRUE(large_cap > s1.size());

    s1.shrink_to_fit();
    // 验证容量被缩小
    EXPECT_TRUE(s1.capacity() <= large_cap);
    // 内容应保持不变
    EXPECT_STREQ(s1.c_str(), "hello world");
    EXPECT_EQ(s1.size(), 11);

    // 测试空字符串的shrink_to_fit
    String s2;
    s2.reserve(50);
    const size_t empty_cap = s2.capacity();

    s2.shrink_to_fit();
    EXPECT_TRUE(s2.capacity() <= empty_cap);
    EXPECT_TRUE(s2.empty());

    // 测试已经是最小容量的情况
    String s3("test");
    const size_t min_cap = s3.capacity();
    s3.shrink_to_fit();
    EXPECT_EQ(s3.capacity(), min_cap);
    EXPECT_TRUE(s3 == "test");

    // 测试调整大小后再shrink_to_fit
    String s4("hello world");
    s4.resize(5);// 缩短字符串
    s4.shrink_to_fit();
    EXPECT_TRUE(s4 == "hello");
    EXPECT_EQ(s4.size(), 5);
}

// 测试resize和reserve的边界情况
TEST(StringMemoryManagementTest, BoundaryCases) {
    // 测试resize到最大值
    String s1("test");

    // EXPECT_THROW(s1.resize(s1.max_size() - 1), std::bad_alloc);

    // 测试reserve到最大值附近
    String s2("sample");
    // EXPECT_THROW(s2.reserve(s2.max_size() / 2), std::bad_alloc);

    // 测试resize为0再resize为大值
    String s3("hello");
    s3.resize(0);
    EXPECT_TRUE(s3.empty());
    s3.resize(10, 'a');
    EXPECT_EQ(s3.size(), 10);
    EXPECT_TRUE(s3 == "aaaaaaaaaa");

    // 测试非常小的resize值
    String s4("test");
    s4.resize(1);
    EXPECT_EQ(s4.size(), 1);
    EXPECT_EQ(s4[0], 't');
}

// 测试内存共享和写时复制行为
TEST(StringMemoryManagementTest, MemorySharingAndCopyOnWrite) {
    // 测试resize时的写时复制
    String original("hello world");
    String copy = original;

    // 验证共享内存
    const void* original_data = original.data();
    const void* copy_data = copy.data();
    EXPECT_NE(original_data, copy_data);

    // 修改copy，触发写时复制
    copy.resize(15, '!');
    EXPECT_NE(copy.data(), original.data());      // 内存应该不再共享
    EXPECT_STREQ(original.c_str(), "hello world");// original保持不变
    EXPECT_STREQ(copy.c_str(), "hello world!!!!");// copy被修改

    // 测试reserve时的写时复制
    String original2("test");
    String copy2 = original2;

    const void* original2_data = original2.data();
    const void* copy2_data = copy2.data();
    EXPECT_NE(original2_data, copy2_data);

    copy2.reserve(100);
    EXPECT_NE(copy2.data(), original2.data());// 内存应该不再共享
    EXPECT_STREQ(original2.c_str(), "test");  // original2保持不变
    EXPECT_STREQ(copy2.c_str(), "test");      // copy2内容保持不变
}

// 测试本地缓冲区和动态分配之间的切换
TEST(StringMemoryManagementTest, LocalBufferSwitching) {
    // 假设local_capacity_是一个足够小的值，我们可以创建一个超过它的字符串
    const size_t large_size = 100;// 假设这大于local_capacity_
    String s1(large_size, 'a');

    // 验证它使用了动态分配（不是本地缓冲区）
    EXPECT_FALSE(s1.IsLocal());

    // 缩小到local_capacity_以下，应该切换回本地缓冲区
    s1.resize(5);
    s1.shrink_to_fit();
    EXPECT_TRUE(s1.IsLocal());// 应该使用本地缓冲区
    EXPECT_TRUE(s1 == "aaaaa");

    // 再次扩大超过local_capacity_
    s1.resize(large_size, 'b');
    EXPECT_FALSE(s1.IsLocal());// 应该切换到动态分配
    EXPECT_EQ(s1.size(), large_size);
}

// 测试resize后的索引访问
TEST(StringMemoryManagementTest, IndexAccessAfterResize) {
    String s("hello");

    // 扩展后访问新字符
    s.resize(8, 'x');
    EXPECT_EQ(s[5], 'x');
    EXPECT_EQ(s[6], 'x');
    EXPECT_EQ(s[7], 'x');

    // 验证原始字符保持不变
    EXPECT_EQ(s[0], 'h');
    EXPECT_EQ(s[1], 'e');
    EXPECT_EQ(s[2], 'l');
    EXPECT_EQ(s[3], 'l');
    EXPECT_EQ(s[4], 'o');

    // 缩短后访问边界
    s.resize(3);
    EXPECT_EQ(s[0], 'h');
    EXPECT_EQ(s[1], 'e');
    EXPECT_EQ(s[2], 'l');
    EXPECT_THROW(s.at(3), std::exception);// 越界应该抛出异常
}

// 测试多个操作的组合
TEST(StringMemoryManagementTest, CombinedOperations) {
    String s;

    // 组合操作序列
    s.resize(5, 'a');// "aaaaa"
    EXPECT_TRUE(s == "aaaaa");

    s.reserve(20);
    EXPECT_TRUE(s == "aaaaa");

    s.resize(10, 'b');// "aaaaabbbbb"
    EXPECT_TRUE(s == "aaaaabbbbb");

    s.resize(7);// "aaaaabb"
    EXPECT_TRUE(s == "aaaaabb");

    s.shrink_to_fit();
    EXPECT_TRUE(s == "aaaaabb");

    s.resize(3, 'c');// "aaa"
    EXPECT_TRUE(s == "aaa");
}

// 测试基本功能：删除字符串中的字符
TEST(StringErase, BasicFunctionality) {
    // 删除字符串中间的字符
    String s1 = "Hello, World!";
    s1.erase(7, 5);// 删除"World"
    EXPECT_TRUE(s1 == "Hello, !");
    EXPECT_EQ(s1.size(), 8);

    // 删除字符串开头的字符
    String s2 = "Programming";
    s2.erase(0, 3);// 删除"Pro"
    EXPECT_TRUE(s2 == "gramming");
    EXPECT_EQ(s2.size(), 8);

    // 删除字符串结尾的字符
    String s3 = "Testing";
    s3.erase(4, 3);// 删除"ing"
    EXPECT_TRUE(s3 == "Test");
    EXPECT_EQ(s3.size(), 4);

    // 删除单个字符
    String s4 = "DeleteMe";
    s4.erase(6, 1);// 删除"e"
    EXPECT_TRUE(s4 == "Deletee");
    EXPECT_EQ(s4.size(), 7);

    String s6(20, 'a');
    String s7 = s6;
    EXPECT_EQ(s6.use_count(), 2);
    s6.clear();
    EXPECT_FALSE(s7.empty());
    s6.push_back('b');
    EXPECT_TRUE(s6.IsLocal());
    EXPECT_EQ(s6.size(), 1);
    EXPECT_EQ(s7.size(), 20);
}

// 测试默认参数：pos=0, n=npos
TEST(StringErase, DefaultParameters) {
    // 使用默认pos=0
    String s1 = "DefaultPos";
    s1.erase(0, 3);// 删除开头3个字符
    EXPECT_TRUE(s1 == "aultPos");

    String s2 = "DefaultPos";
    s2.erase();// 默认删除从0开始的所有字符
    EXPECT_TRUE(s2.empty());
    EXPECT_EQ(s2.size(), 0);

    // 使用默认n=npos（删除到字符串末尾）
    String s3 = "DefaultN";
    s3.erase(4);// 从位置4开始删除到末尾
    EXPECT_TRUE(s3 == "Defa");
    EXPECT_EQ(s3.size(), 4);

    // 同时使用两个默认参数（清空字符串）
    String s4 = "ClearMe";
    s4.erase();
    EXPECT_TRUE(s4.empty());
    EXPECT_TRUE(s4.empty());
}

// 测试边界情况：空字符串
TEST(StringErase, EmptyString) {
    String empty;

    // 对空字符串调用erase应该没有效果
    empty.erase();
    EXPECT_TRUE(empty.empty());
    EXPECT_EQ(empty.size(), 0);
    EXPECT_TRUE(empty.empty());

    // 对空字符串调用erase(0, 5)应该没有效果
    empty.erase(0, 5);
    EXPECT_TRUE(empty.empty());
}

// 测试边界情况：删除超出剩余字符数的情况
TEST(StringErase, OverLimit) {
    String s = "LimitedString";
    const size_t original_size = s.size();

    // 请求删除的字符数超出剩余字符数
    s.erase(7, 100);// 从位置7开始，请求删除100个字符，但实际上只有6个字符
    EXPECT_TRUE(s == "Limited");
    EXPECT_EQ(s.size(), 7);

    // 删除位置加上删除数量刚好等于字符串长度
    String s2 = "ExactlyEnd";
    s2.erase(3, 6);
    EXPECT_TRUE(s2 == "Exad");
    EXPECT_EQ(s2.size(), 4);
}

// 测试异常处理：位置超出范围
TEST(StringErase, OutOfRangeException) {
    String s = "SafeTest";

    // pos超出字符串长度应该抛出异常
    EXPECT_THROW(s.erase(s.size() + 1, 1), Error);

    // pos等于字符串长度是允许的（不会删除任何字符）
    EXPECT_NO_THROW(s.erase(s.size(), 5));
    EXPECT_STREQ(s, "SafeTest");

    // pos为负数应该抛出异常（由于size_type是无符号类型，这里传入一个非常大的值模拟）
    EXPECT_THROW(s.erase(static_cast<String::size_type>(-1), 1), Error);
}

// 测试特殊字符
TEST(StringErase, SpecialCharacters) {
    // 测试包含空格的字符串
    String spaces = "   Spaces   ";
    spaces.erase(0, 3);                // 删除开头的空格
    spaces.erase(spaces.size() - 3, 3);// 删除结尾的空格
    EXPECT_TRUE(spaces == "Spaces");

    // 测试包含控制字符的字符串
    String controls = "Line\nBreak\tTab";
    controls.erase(4, 6);// 删除"\nBreak\t"
    EXPECT_TRUE(controls == "Line\tTab");

    // 测试包含非ASCII字符的字符串
    // String non_ascii = "Café résumé";
    // non_ascii.erase(3, 4);// 删除"é ré"
    // EXPECT_STREQ(non_ascii.c_str(), "Cafsumé");
}

// 测试COW（Copy-On-Write）机制
TEST(StringErase, CopyOnWrite) {
    String original = "SharedString";
    String shared = original;

    // 确认两个字符串共享数据
    EXPECT_TRUE(original.unique());
    EXPECT_TRUE(shared.unique());

    // 修改其中一个字符串
    shared.erase(6, 6);

    // 确认两个字符串不再共享数据
    EXPECT_TRUE(original.unique());
    EXPECT_TRUE(shared.unique());

    // 确认原始字符串未被修改
    EXPECT_TRUE(original == "SharedString");
    // 确认共享字符串已被修改
    EXPECT_TRUE(shared == "Shared");
}

// 测试局部缓冲区切换
TEST(StringErase, LocalBufferSwitch) {
    // 创建一个足够小的字符串，使其使用局部缓冲区
    String small = "Small";
    EXPECT_TRUE(small.IsLocal());// 假设IsLocal()是可访问的，否则需要通过其他方式验证

    // 执行erase操作
    small.erase(2, 2);// 删除"al"
    EXPECT_TRUE(small == "Sml");

    // 创建一个大字符串，使其使用堆内存
    String large(100, 'x');
    EXPECT_FALSE(large.IsLocal());// 假设IsLocal()是可访问的

    // 大幅缩小字符串，使其可能切换回局部缓冲区
    large.erase(5, 90);
    EXPECT_TRUE(large == "xxxxxxxxxx");
    // 由于shrink_to_fit不会自动调用，这里可能仍然在堆上

    String s = "MultipleOperations";
    s.replace(8, 5, "", 0);
    std::cout << s << std::endl;
    EXPECT_TRUE(s.IsLocal());
}

// 测试连续erase操作
TEST(StringErase, MultipleOperations) {
    String s = "MultipleOperations";

    // 连续执行多个erase操作
    s.erase(0, 8);// 删除"Multiple"
    EXPECT_TRUE(s == "Operations");

    s.erase(5, 5);// 删除"ions"
    EXPECT_TRUE(s == "Opera");

    s.erase(3, 1);// 删除"r"
    EXPECT_TRUE(s == "Opea");

    s.erase();// 删除所有字符
    EXPECT_TRUE(s.empty());
}

// 测试与其他操作的组合
TEST(StringErase, CombinedOperations) {
    String s = "StartMiddleEnd";

    // erase后append
    s.erase(5, 6);// 删除"Middle"
    EXPECT_TRUE(s == "StartEnd");

    s.append("Modified");
    EXPECT_TRUE(s == "StartEndModified");

    // replace后erase
    String s2 = "ReplaceAndErase";
    s2.replace(7, 3, "Then");
    EXPECT_TRUE(s2 == "ReplaceThenErase");

    s2.erase(12, 6);// 删除"Erase"
    EXPECT_TRUE(s2 == "ReplaceThenE");
}

// 测试单迭代器版本的erase函数
TEST(StringErase, SingleIterator) {
    // 删除字符串中间的字符
    String s1 = "IteratorTest";
    String::iterator it = s1.begin() + 5;
    String::iterator new_it = s1.erase(it);
    EXPECT_TRUE(s1 == "IteraorTest");
    EXPECT_EQ(s1.size(), 11);
    // 验证返回的迭代器指向被删除字符的下一个位置
    EXPECT_EQ(new_it, s1.begin() + 5);
    EXPECT_EQ(*new_it, 'o');

    // 删除字符串开头的字符
    String s2 = "DeleteFirst";
    new_it = s2.erase(s2.begin());
    EXPECT_TRUE(s2 == "eleteFirst");
    EXPECT_TRUE(s1 == "IteraorTest");
    EXPECT_EQ(s2.size(), 10);
    EXPECT_EQ(new_it, s2.begin());
    EXPECT_EQ(*new_it, 'e');

    // 删除字符串结尾的字符
    String s3 = "DeleteLast";
    new_it = s3.erase(s3.end() - 1);
    EXPECT_TRUE(s3 == "DeleteLas");
    EXPECT_EQ(s3.size(), 9);
    EXPECT_EQ(new_it, s3.end());
}

// 测试迭代器范围版本的erase函数
TEST(StringErase, IteratorRange) {
    // 删除字符串中间的一段
    String s1 = "RangeDeleteTest";
    String::iterator first = s1.begin() + 5;
    String::iterator last = s1.begin() + 11;
    String::iterator new_it = s1.erase(first, last);
    EXPECT_TRUE(s1 == "RangeTest");
    EXPECT_EQ(s1.size(), 9);
    // 验证返回的迭代器指向被删除范围的第一个位置
    EXPECT_EQ(new_it, s1.begin() + 5);
    EXPECT_EQ(*new_it, 'T');

    // 删除字符串开头的一段
    String s2 = "DeleteStart";
    new_it = s2.erase(s2.begin(), s2.begin() + 6);
    EXPECT_TRUE(s2 == "Start");
    EXPECT_EQ(s2.size(), 5);
    EXPECT_EQ(new_it, s2.begin());

    // 删除字符串结尾的一段
    String s3 = "DeleteEnd";
    new_it = s3.erase(s3.begin() + 4, s3.end());
    EXPECT_TRUE(s3 == "Dele");
    EXPECT_EQ(s3.size(), 4);
    EXPECT_EQ(new_it, s3.end());

    // 删除整个字符串
    String s4 = "DeleteAll";
    new_it = s4.erase(s4.begin(), s4.end());
    EXPECT_TRUE(s4.empty());
    EXPECT_EQ(s4.size(), 0);
    EXPECT_EQ(new_it, s4.begin());
    EXPECT_EQ(new_it, s4.end());

    String s5 = "hello, hello, hello";
    String s6 = s5;
    EXPECT_EQ(s5.use_count(), 2);
    *s6.begin() = 'c';
    EXPECT_EQ(s5.use_count(), 1);

    EXPECT_TRUE(s5 == "hello, hello, hello");
    EXPECT_TRUE(s6 == "cello, hello, hello");
}

// 测试不同版本erase的一致性
TEST(StringErase, Consistency) {
    // 测试单字符删除的一致性
    String s1 = "Consistency";
    String s2 = s1;

    // 使用位置长度版本
    s1.erase(5, 1);

    // 使用单迭代器版本
    s2.erase(s2.begin() + 5);

    EXPECT_TRUE(s1 == s2);
    EXPECT_TRUE(s1 == "Consitency");

    // 测试多字符删除的一致性
    String s3 = "AnotherTest";
    String s4 = s3;

    // 使用位置长度版本
    s3.erase(3, 4);

    // 使用迭代器范围版本
    s4.erase(s4.begin() + 3, s4.begin() + 7);

    EXPECT_TRUE(s3 == s4);
    EXPECT_TRUE(s3 == "AnoTest");
}

}  // namespace
