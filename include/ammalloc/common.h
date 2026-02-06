//
// Created by richard on 2/6/26.
//

#ifndef AETHERMIND_MALLOC_COMMON_H
#define AETHERMIND_MALLOC_COMMON_H

#include <cstddef>

namespace aethermind {

namespace details {

/**
 * @brief 解析带单位的内存大小字符串
 *
 * 支持格式示例：
 * - "1024"      -> 1024
 * - "64KB"      -> 64 * 1024
 * - "16 M"      -> 16 * 1024 * 1024 (允许空格)
 * - "1gb"       -> 1 * 1024 * 1024 * 1024 (忽略大小写)
 *
 * @param str 环境变量字符串
 * @return size_t 解析后的字节数。解析失败返回 0。
 */
size_t ParseSize(const char* str);

/**
 * @brief 解析环境变量中的布尔值
 *
 * 支持的 Truthy 值 (不区分大小写，忽略首尾空格):
 * - "1"
 * - "true"
 * - "on"
 * - "yes"
 *
 * 其他所有值均返回 false。
 */
bool ParseBool(const char* str);

}// namespace details

}// namespace aethermind

#endif//AETHERMIND_MALLOC_COMMON_H
