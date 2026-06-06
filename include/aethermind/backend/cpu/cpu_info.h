//
// Created by richard on 6/6/26.
//

#ifndef AETHERMIND_CPU_INFO_H
#define AETHERMIND_CPU_INFO_H

namespace aethermind {
namespace cpu {

struct CpuFeatures {
    // x86 架构族
    bool has_sse4_1 = false;
    bool has_avx2 = false;
    bool has_avx512f = false;
    bool has_vnni = false;// INT8 矩阵乘法加速
    bool has_amx = false; // Intel 先进矩阵扩展

    // ARM 架构族
    bool has_neon = false;
    bool has_sve = false;
    bool has_dotprod = false;// ARM 的 INT8 加速
};

// 全局唯一的获取接口 (单例)
const CpuFeatures& GetCpuFeatures() noexcept;

}// namespace cpu
}// namespace aethermind

#endif// AETHERMIND_CPU_INFO_H
