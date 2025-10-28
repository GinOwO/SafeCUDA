/**
 * @file ptx_modifier.cpp
 * @brief Implementation of PTX modification engine
 *
 * Parses PTX assembly files, identifies memory operations using pattern
 * matching, and injects SafeCUDA bounds checking macros.
 *
 * @author Kiran <kiran.pdas2022@vitstudent.ac.in>
 * @date 2025-08-12
 * @version 1.1.0
 * @copyright Copyright (c) 2025 SafeCUDA Project. Licensed under GPL v3.
 *
 * Change Log:
 * - 2025-10-23: Implemented __bounds_check_safecuda and
 *		__bounds_check_safecuda_no_trap for fail fast
 * - 2025-09-23: Now inserts bounds_check directly into every ptx file
 * - 2025-09-22: Initial Implementation
 * - 2025-08-12: Initial File
 */

#include "ptx_modifier.h"

#include "pattern_matcher.h"

#include <fstream>
#include <iostream>

namespace sf_nvcc = safecuda::tools::sf_nvcc;
namespace fs = std::filesystem;

#ifdef NDEBUG
std::string trap_ver = R"ptx(
.extern .func  (.param .b32 func_retval0) vprintf
(
	.param .b64 vprintf_param_0,
	.param .b64 vprintf_param_1
)
;
.global .align 1 .u8 FREED_MEM_DEV;
.const .align 8 .u64 d_table;
.global .align 1 .b8 $str[6] = {72, 101, 114, 101, 10};

.func __bounds_check_safecuda(
	.param .b64 __bounds_check_safecuda_param_0
)
{
	.reg .pred 	%p<22>;
	.reg .b16 	%rs<15>;
	.reg .b32 	%r<29>;
	.reg .b64 	%rd<35>;


	ld.param.u64 	%rd12, [__bounds_check_safecuda_param_0];
	mov.u64 	%rd13, $str;
	cvta.global.u64 	%rd14, %rd13;
	{ // callseq 0, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd14;
	.param .b64 param1;
	st.param.b64 	[param1+0], 0;
	.param .b32 retval0;
	call.uni (retval0),
	vprintf,
	(
	param0,
	param1
	);
	ld.param.b32 	%r9, [retval0+0];
	} // callseq 0
	ld.global.u8 	%rs5, [FREED_MEM_DEV];
	setp.ne.s16 	%p1, %rs5, 0;
	@%p1 bra 	$L__BB0_12;

	add.s64 	%rd1, %rd12, -16;
	ld.u16 	%rs6, [%rd12+-16];
	setp.ne.s16 	%p2, %rs6, 23294;
	@%p2 bra 	$L__BB0_12;

	ld.u64 	%rd2, [%rd1+8];
	setp.eq.s64 	%p3, %rd2, 0;
	@%p3 bra 	$L__BB0_12;

	ld.u64 	%rd4, [%rd2];
	setp.gt.u64 	%p4, %rd4, %rd12;
	@%p4 bra 	$L__BB0_9;

	ld.u32 	%rd15, [%rd2+8];
	add.s64 	%rd16, %rd4, %rd15;
	setp.le.u64 	%p5, %rd16, %rd12;
	@%p5 bra 	$L__BB0_9;

	ld.u8 	%rs7, [%rd2+12];
	and.b16  	%rs8, %rs7, 2;
	setp.eq.s16 	%p6, %rs8, 0;
	@%p6 bra 	$L__BB0_26;

	ld.const.u64 	%rd17, [d_table];
	cvta.to.global.u64 	%rd5, %rd17;
	ld.global.u64 	%rd18, [%rd5];
	add.s64 	%rd19, %rd18, 12;
	atom.or.b32 	%r10, [%rd19], 2;
	and.b32  	%r11, %r10, 2;
	setp.ne.s32 	%p7, %r11, 0;
	@%p7 bra 	$L__BB0_8;

	ld.global.u64 	%rd20, [%rd5];
	st.u64 	[%rd20], %rd12;

$L__BB0_8:
	// begin inline asm
	trap;
	// end inline asm
	bra.uni 	$L__BB0_26;

$L__BB0_12:
	ld.const.u64 	%rd25, [d_table];
	cvta.to.global.u64 	%rd26, %rd25;
	add.s64 	%rd7, %rd26, 8;
	ld.global.u32 	%r1, [%rd26+8];
	setp.lt.u32 	%p11, %r1, 2;
	ld.global.u64 	%rd8, [%rd26];
	mov.u32 	%r27, -1;
	mov.u16 	%rs13, 0;
	@%p11 bra 	$L__BB0_19;

	mov.u16 	%rs13, 0;
	mov.u32 	%r27, -1;
	mov.u32 	%r25, 1;

$L__BB0_14:
	cvt.u64.u32 	%rd9, %r25;
	mul.wide.u32 	%rd27, %r25, 24;
	add.s64 	%rd10, %rd8, %rd27;
	ld.u64 	%rd11, [%rd10];
	setp.gt.u64 	%p12, %rd11, %rd12;
	@%p12 bra 	$L__BB0_18;

	ld.u32 	%rd28, [%rd10+8];
	add.s64 	%rd29, %rd11, %rd28;
	setp.le.u64 	%p13, %rd29, %rd12;
	@%p13 bra 	$L__BB0_18;

	ld.u32 	%r4, [%rd10+12];
	setp.eq.s32 	%p14, %r4, 0;
	@%p14 bra 	$L__BB0_26;

	cvt.u32.u64 	%r17, %rd9;
	shr.u32 	%r18, %r4, 1;
	and.b32  	%r19, %r18, 1;
	setp.eq.b32 	%p15, %r19, 1;
	selp.b16 	%rs13, 1, %rs13, %p15;
	selp.b32 	%r27, %r17, %r27, %p15;

$L__BB0_18:
	cvt.u32.u64 	%r20, %rd9;
	add.s32 	%r25, %r20, 1;
	setp.lt.u32 	%p16, %r25, %r1;
	@%p16 bra 	$L__BB0_14;

$L__BB0_19:
	and.b16  	%rs11, %rs13, 255;
	setp.eq.s16 	%p17, %rs11, 0;
	@%p17 bra 	$L__BB0_23;

	add.s64 	%rd30, %rd8, 12;
	atom.or.b32 	%r21, [%rd30], 2;
	and.b32  	%r22, %r21, 2;
	setp.ne.s32 	%p18, %r22, 0;
	@%p18 bra 	$L__BB0_22;

	ld.global.u64 	%rd31, [%rd7+-8];
	st.u64 	[%rd31], %rd12;
	ld.global.u64 	%rd32, [%rd7+-8];
	st.u32 	[%rd32+8], %r27;

$L__BB0_22:
	// begin inline asm
	trap;
	// end inline asm
	bra.uni 	$L__BB0_26;

$L__BB0_23:
	add.s64 	%rd33, %rd8, 12;
	atom.or.b32 	%r23, [%rd33], 1;
	and.b32  	%r24, %r23, 1;
	setp.eq.b32 	%p19, %r24, 1;
	mov.pred 	%p20, 0;
	xor.pred  	%p21, %p19, %p20;
	@%p21 bra 	$L__BB0_25;

	ld.global.u64 	%rd34, [%rd7+-8];
	st.u64 	[%rd34], %rd12;

$L__BB0_25:
	// begin inline asm
	trap;
	// end inline asm
	bra.uni 	$L__BB0_26;

$L__BB0_9:
	ld.const.u64 	%rd21, [d_table];
	cvta.to.global.u64 	%rd6, %rd21;
	ld.global.u64 	%rd22, [%rd6];
	add.s64 	%rd23, %rd22, 12;
	atom.or.b32 	%r12, [%rd23], 1;
	and.b32  	%r13, %r12, 1;
	setp.eq.b32 	%p8, %r13, 1;
	mov.pred 	%p9, 0;
	xor.pred  	%p10, %p8, %p9;
	@%p10 bra 	$L__BB0_11;

	ld.global.u64 	%rd24, [%rd6];
	st.u64 	[%rd24], %rd12;

$L__BB0_11:
	// begin inline asm
	trap;
	// end inline asm

$L__BB0_26:
	ret;

}

)ptx";

std::string no_trap_ver = R"ptx(
.extern .func  (.param .b32 func_retval0) vprintf
(
	.param .b64 vprintf_param_0,
	.param .b64 vprintf_param_1
)
;
.global .align 1 .u8 FREED_MEM_DEV;
.const .align 8 .u64 d_table;
.global .align 1 .b8 $str[6] = {72, 101, 114, 101, 10};

.func __bounds_check_safecuda_no_trap(
	.param .b64 __bounds_check_safecuda_no_trap_param_0
)
{
	.reg .pred 	%p<21>;
	.reg .b16 	%rs<14>;
	.reg .b32 	%r<29>;
	.reg .b64 	%rd<35>;


	ld.param.u64 	%rd12, [__bounds_check_safecuda_no_trap_param_0];
	mov.u64 	%rd13, $str;
	cvta.global.u64 	%rd14, %rd13;
	{ // callseq 1, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd14;
	.param .b64 param1;
	st.param.b64 	[param1+0], 0;
	.param .b32 retval0;
	call.uni (retval0),
	vprintf,
	(
	param0,
	param1
	);
	ld.param.b32 	%r9, [retval0+0];
	} // callseq 1
	add.s64 	%rd1, %rd12, -16;
	ld.u16 	%rs5, [%rd12+-16];
	setp.ne.s16 	%p1, %rs5, 23294;
	@%p1 bra 	$L__BB1_9;

	ld.u64 	%rd2, [%rd1+8];
	setp.eq.s64 	%p2, %rd2, 0;
	@%p2 bra 	$L__BB1_9;

	ld.u64 	%rd4, [%rd2];
	setp.gt.u64 	%p3, %rd4, %rd12;
	@%p3 bra 	$L__BB1_7;

	ld.u32 	%rd15, [%rd2+8];
	add.s64 	%rd16, %rd4, %rd15;
	setp.le.u64 	%p4, %rd16, %rd12;
	@%p4 bra 	$L__BB1_7;

	ld.u8 	%rs6, [%rd2+12];
	and.b16  	%rs7, %rs6, 2;
	setp.eq.s16 	%p5, %rs7, 0;
	@%p5 bra 	$L__BB1_21;

	ld.const.u64 	%rd17, [d_table];
	cvta.to.global.u64 	%rd5, %rd17;
	ld.global.u64 	%rd18, [%rd5];
	add.s64 	%rd19, %rd18, 12;
	atom.or.b32 	%r10, [%rd19], 2;
	and.b32  	%r11, %r10, 2;
	setp.ne.s32 	%p6, %r11, 0;
	@%p6 bra 	$L__BB1_21;

	ld.global.u64 	%rd20, [%rd5];
	st.u64 	[%rd20], %rd12;
	bra.uni 	$L__BB1_21;

$L__BB1_9:
	ld.const.u64 	%rd25, [d_table];
	cvta.to.global.u64 	%rd26, %rd25;
	add.s64 	%rd7, %rd26, 8;
	ld.global.u32 	%r1, [%rd26+8];
	setp.lt.u32 	%p10, %r1, 2;
	ld.global.u64 	%rd8, [%rd26];
	mov.u32 	%r27, -1;
	mov.u16 	%rs12, 0;
	@%p10 bra 	$L__BB1_16;

	mov.u16 	%rs12, 0;
	mov.u32 	%r27, -1;
	mov.u32 	%r25, 1;

$L__BB1_11:
	cvt.u64.u32 	%rd9, %r25;
	mul.wide.u32 	%rd27, %r25, 24;
	add.s64 	%rd10, %rd8, %rd27;
	ld.u64 	%rd11, [%rd10];
	setp.gt.u64 	%p11, %rd11, %rd12;
	@%p11 bra 	$L__BB1_15;

	ld.u32 	%rd28, [%rd10+8];
	add.s64 	%rd29, %rd11, %rd28;
	setp.le.u64 	%p12, %rd29, %rd12;
	@%p12 bra 	$L__BB1_15;

	ld.u32 	%r4, [%rd10+12];
	setp.eq.s32 	%p13, %r4, 0;
	@%p13 bra 	$L__BB1_21;

	cvt.u32.u64 	%r17, %rd9;
	shr.u32 	%r18, %r4, 1;
	and.b32  	%r19, %r18, 1;
	setp.eq.b32 	%p14, %r19, 1;
	selp.b16 	%rs12, 1, %rs12, %p14;
	selp.b32 	%r27, %r17, %r27, %p14;

$L__BB1_15:
	cvt.u32.u64 	%r20, %rd9;
	add.s32 	%r25, %r20, 1;
	setp.lt.u32 	%p15, %r25, %r1;
	@%p15 bra 	$L__BB1_11;

$L__BB1_16:
	and.b16  	%rs10, %rs12, 255;
	setp.eq.s16 	%p16, %rs10, 0;
	@%p16 bra 	$L__BB1_19;

	add.s64 	%rd30, %rd8, 12;
	atom.or.b32 	%r21, [%rd30], 2;
	and.b32  	%r22, %r21, 2;
	setp.ne.s32 	%p17, %r22, 0;
	@%p17 bra 	$L__BB1_21;

	ld.global.u64 	%rd31, [%rd7+-8];
	st.u64 	[%rd31], %rd12;
	ld.global.u64 	%rd32, [%rd7+-8];
	st.u32 	[%rd32+8], %r27;
	bra.uni 	$L__BB1_21;

$L__BB1_7:
	ld.const.u64 	%rd21, [d_table];
	cvta.to.global.u64 	%rd6, %rd21;
	ld.global.u64 	%rd22, [%rd6];
	add.s64 	%rd23, %rd22, 12;
	atom.or.b32 	%r12, [%rd23], 1;
	and.b32  	%r13, %r12, 1;
	setp.eq.b32 	%p7, %r13, 1;
	mov.pred 	%p8, 0;
	xor.pred  	%p9, %p7, %p8;
	@%p9 bra 	$L__BB1_21;

	ld.global.u64 	%rd24, [%rd6];
	st.u64 	[%rd24], %rd12;
	bra.uni 	$L__BB1_21;

$L__BB1_19:
	add.s64 	%rd33, %rd8, 12;
	atom.or.b32 	%r23, [%rd33], 1;
	and.b32  	%r24, %r23, 1;
	setp.eq.b32 	%p18, %r24, 1;
	mov.pred 	%p19, 0;
	xor.pred  	%p20, %p18, %p19;
	@%p20 bra 	$L__BB1_21;

	ld.global.u64 	%rd34, [%rd7+-8];
	st.u64 	[%rd34], %rd12;

$L__BB1_21:
	ret;

}

)ptx";

#else

std::string trap_ver = R"ptx(
.extern .func  (.param .b32 func_retval0) vprintf
(
	.param .b64 vprintf_param_0,
	.param .b64 vprintf_param_1
)
;
.func __trap
()
;
.func  (.param .b32 func_retval0) __uAtomicOr
(
	.param .b64 __uAtomicOr_param_0,
	.param .b32 __uAtomicOr_param_1
)
;
.global .align 1 .u8 FREED_MEM_DEV;
.const .align 8 .u64 d_table;
.global .align 1 .b8 $str[6] = {72, 101, 114, 101, 10};

.func  (.param .b32 func_retval0) _ZN43_INTERNAL_b1f57ba6_12_safecache_cu_237960b88atomicOrEPjj(
	.param .b64 _ZN43_INTERNAL_b1f57ba6_12_safecache_cu_237960b88atomicOrEPjj_param_0,
	.param .b32 _ZN43_INTERNAL_b1f57ba6_12_safecache_cu_237960b88atomicOrEPjj_param_1
)
{
	.reg .b32 	%r<4>;
	.reg .b64 	%rd<3>;
	.loc	7 185 0
$L__func_begin0:
	.loc	7 185 0


	ld.param.u64 	%rd1, [_ZN43_INTERNAL_b1f57ba6_12_safecache_cu_237960b88atomicOrEPjj_param_0];
	ld.param.u32 	%r1, [_ZN43_INTERNAL_b1f57ba6_12_safecache_cu_237960b88atomicOrEPjj_param_1];
$L__tmp0:
	.loc	7 187 3
	mov.b64 	%rd2, %rd1;
	mov.b32 	%r2, %r1;
	{ // callseq 0, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd2;
	.param .b32 param1;
	st.param.b32 	[param1+0], %r2;
	.param .b32 retval0;
	call.uni (retval0),
	__uAtomicOr,
	(
	param0,
	param1
	);
	ld.param.b32 	%r3, [retval0+0];
	} // callseq 0
	st.param.b32 	[func_retval0+0], %r3;
	ret;
$L__tmp1:
$L__func_end0:

}

.func __bounds_check_safecuda(
	.param .b64 __bounds_check_safecuda_param_0
)
{
	.local .align 8 .b8 	__local_depot1[8];
	.reg .b64 	%SP;
	.reg .b64 	%SPL;
	.reg .pred 	%p<43>;
	.reg .b16 	%rs<15>;
	.reg .b32 	%r<33>;
	.reg .b64 	%rd<72>;
	.loc	5 34 0
$L__func_begin1:
	.loc	5 34 0


	mov.u64 	%SPL, __local_depot1;
	cvta.local.u64 	%SP, %SPL;
	ld.param.u64 	%rd5, [__bounds_check_safecuda_param_0];
$L__tmp2:
	.loc	5 36 2
	mov.u64 	%rd6, $str;
	cvta.global.u64 	%rd7, %rd6;
	{ // callseq 1, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd7;
	.param .b64 param1;
	st.param.b64 	[param1+0], 0;
	.param .b32 retval0;
	call.uni (retval0),
	vprintf,
	(
	param0,
	param1
	);
	ld.param.b32 	%r9, [retval0+0];
	} // callseq 1
	.loc	5 38 3
	add.s64 	%rd1, %rd5, -16;
$L__tmp3:
	.loc	5 40 2
	mov.u64 	%rd8, FREED_MEM_DEV;
	cvta.global.u64 	%rd9, %rd8;
	ld.u8 	%rs6, [%rd9];
	and.b16  	%rs7, %rs6, 255;
	setp.ne.s16 	%p8, %rs7, 0;
	not.pred 	%p9, %p8;
	mov.pred 	%p7, 0;
	not.pred 	%p10, %p9;
	mov.pred 	%p40, %p7;
	@%p10 bra 	$L__BB1_2;
	bra.uni 	$L__BB1_1;

$L__BB1_1:
	ld.u16 	%rs8, [%rd1];
	cvt.u32.u16 	%r10, %rs8;
	setp.eq.s32 	%p1, %r10, 23294;
	mov.pred 	%p40, %p1;
	bra.uni 	$L__BB1_2;

$L__BB1_2:
	mov.pred 	%p2, %p40;
	not.pred 	%p11, %p2;
	@%p11 bra 	$L__BB1_16;
	bra.uni 	$L__BB1_3;

$L__BB1_3:
$L__tmp4:
	.loc	5 41 3
	ld.u64 	%rd10, [%rd1+8];
	mov.b64 	%rd11, %rd10;
	st.u64 	[%SP+0], %rd11;
	.loc	5 42 3
	ld.u64 	%rd12, [%SP+0];
	setp.eq.s64 	%p12, %rd12, 0;
	not.pred 	%p13, %p12;
	@%p13 bra 	$L__BB1_5;
	bra.uni 	$L__BB1_4;

$L__BB1_4:
$L__tmp5:
	.loc	5 43 4
	bra.uni 	$L__BB1_17;
$L__tmp6:

$L__BB1_5:
	.loc	5 46 3
	ld.u64 	%rd13, [%SP+0];
	ld.u64 	%rd14, [%rd13];
	setp.ge.u64 	%p15, %rd5, %rd14;
	mov.pred 	%p14, 0;
	not.pred 	%p16, %p15;
	mov.pred 	%p41, %p14;
	@%p16 bra 	$L__BB1_7;
	bra.uni 	$L__BB1_6;

$L__BB1_6:
	.loc	5 47 7
	ld.u64 	%rd15, [%SP+0];
	ld.u64 	%rd16, [%rd15];
	ld.u64 	%rd17, [%SP+0];
	ld.u32 	%r11, [%rd17+8];
	cvt.u64.u32 	%rd18, %r11;
	add.s64 	%rd19, %rd16, %rd18;
	setp.lt.u64 	%p3, %rd5, %rd19;
	mov.pred 	%p41, %p3;
	bra.uni 	$L__BB1_7;

$L__BB1_7:
	mov.pred 	%p4, %p41;
	not.pred 	%p17, %p4;
	@%p17 bra 	$L__BB1_13;
	bra.uni 	$L__BB1_8;

$L__BB1_8:
$L__tmp7:
	.loc	5 49 4
	ld.u64 	%rd29, [%SP+0];
	ld.u32 	%r14, [%rd29+12];
	.loc	5 50 8
	and.b32  	%r15, %r14, 2;
	setp.ne.s32 	%p20, %r15, 0;
	not.pred 	%p21, %p20;
	@%p21 bra 	$L__BB1_12;
	bra.uni 	$L__BB1_9;

$L__BB1_9:
$L__tmp8:
	.loc	5 52 6
	mov.u64 	%rd30, d_table;
	cvta.const.u64 	%rd31, %rd30;
	ld.u64 	%rd32, [%rd31];
	ld.u64 	%rd33, [%rd32];
	add.s64 	%rd34, %rd33, 12;
	.loc	5 51 22
	{ // callseq 4, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd34;
	.param .b32 param1;
	st.param.b32 	[param1+0], 2;
	.param .b32 retval0;
	call.uni (retval0),
	_ZN43_INTERNAL_b1f57ba6_12_safecache_cu_237960b88atomicOrEPjj,
	(
	param0,
	param1
	);
	ld.param.b32 	%r16, [retval0+0];
$L__tmp9:
	} // callseq 4
	.loc	5 55 10
	and.b32  	%r17, %r16, 2;
	.loc	5 56 9
	setp.eq.s32 	%p22, %r17, 0;
	not.pred 	%p23, %p22;
	@%p23 bra 	$L__BB1_11;
	bra.uni 	$L__BB1_10;

$L__BB1_10:
$L__tmp10:
	.loc	5 57 6
	mov.u64 	%rd35, d_table;
	cvta.const.u64 	%rd36, %rd35;
	ld.u64 	%rd37, [%rd36];
	ld.u64 	%rd38, [%rd37];
	st.u64 	[%rd38], %rd5;
	bra.uni 	$L__BB1_11;
$L__tmp11:

$L__BB1_11:
	.loc	5 59 5
	{ // callseq 5, 0
	.reg .b32 temp_param_reg;
	call.uni
	__trap,
	(
	);
	} // callseq 5
	bra.uni 	$L__BB1_12;
$L__tmp12:

$L__BB1_12:
	.loc	5 62 4
	bra.uni 	$L__BB1_36;
$L__tmp13:

$L__BB1_13:
	.loc	5 66 4
	mov.u64 	%rd20, d_table;
	cvta.const.u64 	%rd21, %rd20;
	ld.u64 	%rd22, [%rd21];
	ld.u64 	%rd23, [%rd22];
	add.s64 	%rd24, %rd23, 12;
	{ // callseq 2, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd24;
	.param .b32 param1;
	st.param.b32 	[param1+0], 1;
	.param .b32 retval0;
	call.uni (retval0),
	_ZN43_INTERNAL_b1f57ba6_12_safecache_cu_237960b88atomicOrEPjj,
	(
	param0,
	param1
	);
	ld.param.b32 	%r12, [retval0+0];
$L__tmp14:
	} // callseq 2
	.loc	5 68 3
	and.b32  	%r13, %r12, 1;
	setp.eq.s32 	%p18, %r13, 0;
	not.pred 	%p19, %p18;
	@%p19 bra 	$L__BB1_15;
	bra.uni 	$L__BB1_14;

$L__BB1_14:
$L__tmp15:
	.loc	5 69 4
	mov.u64 	%rd25, d_table;
	cvta.const.u64 	%rd26, %rd25;
	ld.u64 	%rd27, [%rd26];
	ld.u64 	%rd28, [%rd27];
	st.u64 	[%rd28], %rd5;
	bra.uni 	$L__BB1_15;
$L__tmp16:

$L__BB1_15:
	.loc	5 71 3
	{ // callseq 3, 0
	.reg .b32 temp_param_reg;
	call.uni
	__trap,
	(
	);
	} // callseq 3
	.loc	5 73 3
	bra.uni 	$L__BB1_36;
$L__tmp17:

$L__BB1_16:
	.loc	5 75 1
	bra.uni 	$L__BB1_17;

$L__BB1_17:
$L__tmp18:
	.loc	5 77 2
	mov.u16 	%rs9, 0;
	mov.b16 	%rs1, %rs9;
$L__tmp19:
	.loc	5 78 2
	mov.u32 	%r18, -1;
	mov.b32 	%r1, %r18;
$L__tmp20:
	.loc	5 80 2
	mov.u32 	%r19, 1;
	mov.b32 	%r2, %r19;
$L__tmp21:
	mov.u16 	%rs12, %rs1;
$L__tmp22:
	mov.u32 	%r29, %r1;
$L__tmp23:
	mov.u32 	%r30, %r2;
$L__tmp24:
	bra.uni 	$L__BB1_18;

$L__BB1_18:
	mov.u32 	%r4, %r30;
	mov.u32 	%r3, %r29;
	mov.u16 	%rs2, %rs12;
$L__tmp25:
	mov.u64 	%rd39, d_table;
$L__tmp26:
	cvta.const.u64 	%rd40, %rd39;
	ld.u64 	%rd41, [%rd40];
	ld.u32 	%r20, [%rd41+8];
	setp.lt.u32 	%p24, %r4, %r20;
	not.pred 	%p25, %p24;
	@%p25 bra 	$L__BB1_29;
	bra.uni 	$L__BB1_19;

$L__BB1_19:
$L__tmp27:
	.loc	5 81 3
	mov.u64 	%rd62, d_table;
	cvta.const.u64 	%rd63, %rd62;
	ld.u64 	%rd64, [%rd63];
	ld.u64 	%rd65, [%rd64];
	cvt.u64.u32 	%rd66, %r4;
	mul.lo.s64 	%rd67, %rd66, 24;
	add.s64 	%rd4, %rd65, %rd67;
$L__tmp28:
	.loc	5 83 3
	ld.u64 	%rd68, [%rd4];
	setp.le.u64 	%p33, %rd68, %rd5;
	mov.pred 	%p32, 0;
	not.pred 	%p34, %p33;
	mov.pred 	%p42, %p32;
	@%p34 bra 	$L__BB1_21;
	bra.uni 	$L__BB1_20;

$L__BB1_20:
	.loc	5 84 7
	ld.u64 	%rd69, [%rd4];
	ld.u32 	%r25, [%rd4+8];
	cvt.u64.u32 	%rd70, %r25;
	add.s64 	%rd71, %rd69, %rd70;
	setp.lt.u64 	%p5, %rd5, %rd71;
	mov.pred 	%p42, %p5;
	bra.uni 	$L__BB1_21;

$L__BB1_21:
	mov.pred 	%p6, %p42;
	not.pred 	%p35, %p6;
	mov.u16 	%rs14, %rs2;
$L__tmp29:
	mov.u32 	%r32, %r3;
$L__tmp30:
	@%p35 bra 	$L__BB1_27;
	bra.uni 	$L__BB1_22;

$L__BB1_22:
$L__tmp31:
	.loc	5 86 4
	ld.u32 	%r26, [%rd4+12];
	setp.eq.s32 	%p36, %r26, 0;
	not.pred 	%p37, %p36;
	@%p37 bra 	$L__BB1_24;
	bra.uni 	$L__BB1_23;

$L__BB1_23:
$L__tmp32:
	.loc	5 87 5
	bra.uni 	$L__BB1_36;
$L__tmp33:

$L__BB1_24:
	.loc	5 89 4
	ld.u32 	%r27, [%rd4+12];
	.loc	5 90 8
	and.b32  	%r28, %r27, 2;
	setp.ne.s32 	%p38, %r28, 0;
	not.pred 	%p39, %p38;
	mov.u16 	%rs13, %rs2;
$L__tmp34:
	mov.u32 	%r31, %r3;
$L__tmp35:
	@%p39 bra 	$L__BB1_26;
	bra.uni 	$L__BB1_25;

$L__BB1_25:
$L__tmp36:
	.loc	5 91 5
	mov.u16 	%rs11, 1;
	mov.b16 	%rs3, %rs11;
$L__tmp37:
	.loc	5 92 5
	mov.b32 	%r5, %r4;
$L__tmp38:
	mov.u16 	%rs13, %rs3;
$L__tmp39:
	mov.u32 	%r31, %r5;
$L__tmp40:
	bra.uni 	$L__BB1_26;

$L__BB1_26:
	mov.u32 	%r6, %r31;
	mov.u16 	%rs4, %rs13;
$L__tmp41:
	mov.u16 	%rs14, %rs4;
$L__tmp42:
	mov.u32 	%r32, %r6;
$L__tmp43:
	bra.uni 	$L__BB1_27;
$L__tmp44:

$L__BB1_27:
	.loc	5 80 48
	mov.u32 	%r7, %r32;
	mov.u16 	%rs5, %rs14;
$L__tmp45:
	bra.uni 	$L__BB1_28;

$L__BB1_28:
	add.s32 	%r8, %r4, 1;
$L__tmp46:
	mov.u16 	%rs12, %rs5;
$L__tmp47:
	mov.u32 	%r29, %r7;
$L__tmp48:
	mov.u32 	%r30, %r8;
$L__tmp49:
	bra.uni 	$L__BB1_18;
$L__tmp50:

$L__BB1_29:
	.loc	5 97 2
	and.b16  	%rs10, %rs2, 255;
	setp.ne.s16 	%p26, %rs10, 0;
	not.pred 	%p27, %p26;
	@%p27 bra 	$L__BB1_33;
	bra.uni 	$L__BB1_30;

$L__BB1_30:
$L__tmp51:
	.loc	5 98 3
	mov.u64 	%rd51, d_table;
	cvta.const.u64 	%rd52, %rd51;
	ld.u64 	%rd53, [%rd52];
	ld.u64 	%rd54, [%rd53];
	add.s64 	%rd55, %rd54, 12;
	.loc	5 98 20
	{ // callseq 8, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd55;
	.param .b32 param1;
	st.param.b32 	[param1+0], 2;
	.param .b32 retval0;
	call.uni (retval0),
	_ZN43_INTERNAL_b1f57ba6_12_safecache_cu_237960b88atomicOrEPjj,
	(
	param0,
	param1
	);
	ld.param.b32 	%r23, [retval0+0];
$L__tmp52:
	} // callseq 8
	.loc	5 100 3
	and.b32  	%r24, %r23, 2;
	setp.eq.s32 	%p30, %r24, 0;
	not.pred 	%p31, %p30;
	@%p31 bra 	$L__BB1_32;
	bra.uni 	$L__BB1_31;

$L__BB1_31:
$L__tmp53:
	.loc	5 101 4
	mov.u64 	%rd56, d_table;
	cvta.const.u64 	%rd57, %rd56;
	ld.u64 	%rd58, [%rd57];
	ld.u64 	%rd59, [%rd58];
	st.u64 	[%rd59], %rd5;
	.loc	5 102 4
	ld.u64 	%rd60, [%rd57];
	ld.u64 	%rd61, [%rd60];
	st.u32 	[%rd61+8], %r3;
	bra.uni 	$L__BB1_32;
$L__tmp54:

$L__BB1_32:
	.loc	5 104 3
	{ // callseq 9, 0
	.reg .b32 temp_param_reg;
	call.uni
	__trap,
	(
	);
	} // callseq 9
	.loc	5 105 3
	bra.uni 	$L__BB1_36;
$L__tmp55:

$L__BB1_33:
	.loc	5 108 2
	mov.u64 	%rd42, d_table;
	cvta.const.u64 	%rd43, %rd42;
	ld.u64 	%rd44, [%rd43];
	ld.u64 	%rd45, [%rd44];
	add.s64 	%rd46, %rd45, 12;
	.loc	5 108 19
	{ // callseq 6, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd46;
	.param .b32 param1;
	st.param.b32 	[param1+0], 1;
	.param .b32 retval0;
	call.uni (retval0),
	_ZN43_INTERNAL_b1f57ba6_12_safecache_cu_237960b88atomicOrEPjj,
	(
	param0,
	param1
	);
	ld.param.b32 	%r21, [retval0+0];
$L__tmp56:
	} // callseq 6
	.loc	5 111 2
	and.b32  	%r22, %r21, 1;
	setp.eq.s32 	%p28, %r22, 0;
	not.pred 	%p29, %p28;
	@%p29 bra 	$L__BB1_35;
	bra.uni 	$L__BB1_34;

$L__BB1_34:
$L__tmp57:
	.loc	5 112 3
	mov.u64 	%rd47, d_table;
	cvta.const.u64 	%rd48, %rd47;
	ld.u64 	%rd49, [%rd48];
	ld.u64 	%rd50, [%rd49];
	st.u64 	[%rd50], %rd5;
	bra.uni 	$L__BB1_35;
$L__tmp58:

$L__BB1_35:
	.loc	5 116 2
	{ // callseq 7, 0
	.reg .b32 temp_param_reg;
	call.uni
	__trap,
	(
	);
	} // callseq 7
	.loc	5 117 1
	bra.uni 	$L__BB1_36;

$L__BB1_36:
	ret;
$L__tmp59:
$L__func_end1:

}
.func __trap()
{



	// begin inline asm
	trap;
	// end inline asm
	ret;
$L__func_end3:

}
.func  (.param .b32 func_retval0) __uAtomicOr(
	.param .b64 __uAtomicOr_param_0,
	.param .b32 __uAtomicOr_param_1
)
{
	.reg .b32 	%r<3>;
	.reg .b64 	%rd<2>;


	ld.param.u64 	%rd1, [__uAtomicOr_param_0];
	ld.param.u32 	%r1, [__uAtomicOr_param_1];
	atom.or.b32 	%r2, [%rd1], %r1;
	st.param.b32 	[func_retval0+0], %r2;
	ret;
$L__func_end4:

}

	.file	1 "/home/gin/Desktop/SafeCUDA/examples/safecache.cuh"
	.file	2 "/usr/include/stdint.h"
	.file	3 "/usr/include/bits/types.h"
	.file	4 "/usr/include/bits/stdint-uintn.h"
	.file	5 "/home/gin/Desktop/SafeCUDA/examples/safecache.cu"
	.file	6 "/usr/include/bits/stdint-intn.h"
	.file	7 "/usr/local/cuda/bin/../targets/x86_64-linux/include/device_atomic_functions.hpp"
	.section	.debug_loc
	{
.b64 $L__tmp19
.b64 $L__tmp22
.b8 6
.b8 0
.b8 144
.b8 177
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp22
.b64 $L__tmp26
.b8 7
.b8 0
.b8 144
.b8 178
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp26
.b64 $L__tmp29
.b8 6
.b8 0
.b8 144
.b8 178
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp29
.b64 $L__tmp34
.b8 7
.b8 0
.b8 144
.b8 180
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp34
.b64 $L__tmp37
.b8 7
.b8 0
.b8 144
.b8 179
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp37
.b64 $L__tmp39
.b8 6
.b8 0
.b8 144
.b8 179
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp39
.b64 $L__tmp41
.b8 7
.b8 0
.b8 144
.b8 179
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp41
.b64 $L__tmp42
.b8 6
.b8 0
.b8 144
.b8 180
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp42
.b64 $L__tmp45
.b8 7
.b8 0
.b8 144
.b8 180
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp45
.b64 $L__tmp47
.b8 6
.b8 0
.b8 144
.b8 181
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp47
.b64 $L__func_end1
.b8 7
.b8 0
.b8 144
.b8 178
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 0
.b64 0
.b64 $L__tmp20
.b64 $L__tmp23
.b8 5
.b8 0
.b8 144
.b8 177
.b8 228
.b8 149
.b8 1
.b64 $L__tmp23
.b64 $L__tmp25
.b8 6
.b8 0
.b8 144
.b8 185
.b8 228
.b8 200
.b8 171
.b8 2
.b64 $L__tmp25
.b64 $L__tmp30
.b8 5
.b8 0
.b8 144
.b8 179
.b8 228
.b8 149
.b8 1
.b64 $L__tmp30
.b64 $L__tmp35
.b8 6
.b8 0
.b8 144
.b8 178
.b8 230
.b8 200
.b8 171
.b8 2
.b64 $L__tmp35
.b64 $L__tmp38
.b8 6
.b8 0
.b8 144
.b8 177
.b8 230
.b8 200
.b8 171
.b8 2
.b64 $L__tmp38
.b64 $L__tmp40
.b8 5
.b8 0
.b8 144
.b8 181
.b8 228
.b8 149
.b8 1
.b64 $L__tmp40
.b64 $L__tmp41
.b8 6
.b8 0
.b8 144
.b8 177
.b8 230
.b8 200
.b8 171
.b8 2
.b64 $L__tmp41
.b64 $L__tmp43
.b8 5
.b8 0
.b8 144
.b8 182
.b8 228
.b8 149
.b8 1
.b64 $L__tmp43
.b64 $L__tmp45
.b8 6
.b8 0
.b8 144
.b8 178
.b8 230
.b8 200
.b8 171
.b8 2
.b64 $L__tmp45
.b64 $L__tmp48
.b8 5
.b8 0
.b8 144
.b8 183
.b8 228
.b8 149
.b8 1
.b64 $L__tmp48
.b64 $L__func_end1
.b8 6
.b8 0
.b8 144
.b8 185
.b8 228
.b8 200
.b8 171
.b8 2
.b64 0
.b64 0
.b64 $L__tmp21
.b64 $L__tmp24
.b8 5
.b8 0
.b8 144
.b8 178
.b8 228
.b8 149
.b8 1
.b64 $L__tmp24
.b64 $L__tmp26
.b8 6
.b8 0
.b8 144
.b8 176
.b8 230
.b8 200
.b8 171
.b8 2
.b64 $L__tmp26
.b64 $L__tmp46
.b8 5
.b8 0
.b8 144
.b8 180
.b8 228
.b8 149
.b8 1
.b64 $L__tmp46
.b64 $L__tmp49
.b8 5
.b8 0
.b8 144
.b8 184
.b8 228
.b8 149
.b8 1
.b64 $L__tmp49
.b64 $L__func_end1
.b8 6
.b8 0
.b8 144
.b8 176
.b8 230
.b8 200
.b8 171
.b8 2
.b64 0
.b64 0
.b64 $L__tmp76
.b64 $L__tmp79
.b8 6
.b8 0
.b8 144
.b8 177
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp79
.b64 $L__tmp83
.b8 7
.b8 0
.b8 144
.b8 176
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp83
.b64 $L__tmp86
.b8 6
.b8 0
.b8 144
.b8 178
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp86
.b64 $L__tmp91
.b8 7
.b8 0
.b8 144
.b8 178
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp91
.b64 $L__tmp94
.b8 7
.b8 0
.b8 144
.b8 177
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp94
.b64 $L__tmp96
.b8 6
.b8 0
.b8 144
.b8 179
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp96
.b64 $L__tmp98
.b8 7
.b8 0
.b8 144
.b8 177
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp98
.b64 $L__tmp99
.b8 6
.b8 0
.b8 144
.b8 180
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp99
.b64 $L__tmp102
.b8 7
.b8 0
.b8 144
.b8 178
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp102
.b64 $L__tmp104
.b8 6
.b8 0
.b8 144
.b8 181
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp104
.b64 $L__func_end2
.b8 7
.b8 0
.b8 144
.b8 176
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 0
.b64 0
.b64 $L__tmp77
.b64 $L__tmp80
.b8 5
.b8 0
.b8 144
.b8 177
.b8 228
.b8 149
.b8 1
.b64 $L__tmp80
.b64 $L__tmp82
.b8 6
.b8 0
.b8 144
.b8 185
.b8 228
.b8 200
.b8 171
.b8 2
.b64 $L__tmp82
.b64 $L__tmp87
.b8 5
.b8 0
.b8 144
.b8 179
.b8 228
.b8 149
.b8 1
.b64 $L__tmp87
.b64 $L__tmp92
.b8 6
.b8 0
.b8 144
.b8 178
.b8 230
.b8 200
.b8 171
.b8 2
.b64 $L__tmp92
.b64 $L__tmp95
.b8 6
.b8 0
.b8 144
.b8 177
.b8 230
.b8 200
.b8 171
.b8 2
.b64 $L__tmp95
.b64 $L__tmp97
.b8 5
.b8 0
.b8 144
.b8 181
.b8 228
.b8 149
.b8 1
.b64 $L__tmp97
.b64 $L__tmp98
.b8 6
.b8 0
.b8 144
.b8 177
.b8 230
.b8 200
.b8 171
.b8 2
.b64 $L__tmp98
.b64 $L__tmp100
.b8 5
.b8 0
.b8 144
.b8 182
.b8 228
.b8 149
.b8 1
.b64 $L__tmp100
.b64 $L__tmp102
.b8 6
.b8 0
.b8 144
.b8 178
.b8 230
.b8 200
.b8 171
.b8 2
.b64 $L__tmp102
.b64 $L__tmp105
.b8 5
.b8 0
.b8 144
.b8 183
.b8 228
.b8 149
.b8 1
.b64 $L__tmp105
.b64 $L__func_end2
.b8 6
.b8 0
.b8 144
.b8 185
.b8 228
.b8 200
.b8 171
.b8 2
.b64 0
.b64 0
.b64 $L__tmp78
.b64 $L__tmp81
.b8 5
.b8 0
.b8 144
.b8 178
.b8 228
.b8 149
.b8 1
.b64 $L__tmp81
.b64 $L__tmp83
.b8 6
.b8 0
.b8 144
.b8 176
.b8 230
.b8 200
.b8 171
.b8 2
.b64 $L__tmp83
.b64 $L__tmp103
.b8 5
.b8 0
.b8 144
.b8 180
.b8 228
.b8 149
.b8 1
.b64 $L__tmp103
.b64 $L__tmp106
.b8 5
.b8 0
.b8 144
.b8 184
.b8 228
.b8 149
.b8 1
.b64 $L__tmp106
.b64 $L__func_end2
.b8 6
.b8 0
.b8 144
.b8 176
.b8 230
.b8 200
.b8 171
.b8 2
.b64 0
.b64 0
	}
	.section	.debug_abbrev
	{
.b8 1
.b8 17
.b8 1
.b8 37
.b8 8
.b8 19
.b8 5
.b8 3
.b8 8
.b8 16
.b8 6
.b8 27
.b8 8
.b8 17
.b8 1
.b8 0
.b8 0
.b8 2
.b8 52
.b8 0
.b8 3
.b8 8
.b8 73
.b8 19
.b8 63
.b8 12
.b8 58
.b8 11
.b8 59
.b8 11
.b8 51
.b8 11
.b8 2
.b8 10
.b8 135,64
.b8 8
.b8 0
.b8 0
.b8 3
.b8 36
.b8 0
.b8 3
.b8 8
.b8 62
.b8 11
.b8 11
.b8 11
.b8 0
.b8 0
.b8 4
.b8 15
.b8 0
.b8 73
.b8 19
.b8 51
.b8 6
.b8 0
.b8 0
.b8 5
.b8 19
.b8 1
.b8 3
.b8 8
.b8 11
.b8 11
.b8 58
.b8 11
.b8 59
.b8 11
.b8 0
.b8 0
.b8 6
.b8 13
.b8 0
.b8 3
.b8 8
.b8 73
.b8 19
.b8 58
.b8 11
.b8 59
.b8 11
.b8 56
.b8 10
.b8 0
.b8 0
.b8 7
.b8 22
.b8 0
.b8 73
.b8 19
.b8 3
.b8 8
.b8 58
.b8 11
.b8 59
.b8 11
.b8 0
.b8 0
.b8 8
.b8 1
.b8 1
.b8 73
.b8 19
.b8 0
.b8 0
.b8 9
.b8 33
.b8 0
.b8 73
.b8 19
.b8 55
.b8 11
.b8 0
.b8 0
.b8 10
.b8 36
.b8 0
.b8 3
.b8 8
.b8 11
.b8 11
.b8 62
.b8 11
.b8 0
.b8 0
.b8 11
.b8 46
.b8 1
.b8 17
.b8 1
.b8 18
.b8 1
.b8 64
.b8 10
.b8 135,64
.b8 8
.b8 3
.b8 8
.b8 58
.b8 11
.b8 59
.b8 11
.b8 73
.b8 19
.b8 0
.b8 0
.b8 12
.b8 5
.b8 0
.b8 2
.b8 10
.b8 51
.b8 11
.b8 3
.b8 8
.b8 58
.b8 11
.b8 59
.b8 11
.b8 73
.b8 19
.b8 0
.b8 0
.b8 13
.b8 46
.b8 1
.b8 17
.b8 1
.b8 18
.b8 1
.b8 64
.b8 10
.b8 135,64
.b8 8
.b8 3
.b8 8
.b8 58
.b8 11
.b8 59
.b8 11
.b8 73
.b8 19
.b8 63
.b8 12
.b8 0
.b8 0
.b8 14
.b8 11
.b8 1
.b8 17
.b8 1
.b8 18
.b8 1
.b8 0
.b8 0
.b8 15
.b8 52
.b8 0
.b8 2
.b8 10
.b8 51
.b8 11
.b8 3
.b8 8
.b8 58
.b8 11
.b8 59
.b8 11
.b8 73
.b8 19
.b8 0
.b8 0
.b8 16
.b8 52
.b8 0
.b8 2
.b8 6
.b8 3
.b8 8
.b8 58
.b8 11
.b8 59
.b8 11
.b8 73
.b8 19
.b8 0
.b8 0
.b8 17
.b8 52
.b8 0
.b8 51
.b8 11
.b8 2
.b8 10
.b8 3
.b8 8
.b8 58
.b8 11
.b8 59
.b8 11
.b8 73
.b8 19
.b8 0
.b8 0
.b8 18
.b8 59
.b8 0
.b8 3
.b8 8
.b8 0
.b8 0
.b8 19
.b8 38
.b8 0
.b8 73
.b8 19
.b8 0
.b8 0
.b8 0
	}
	.section	.debug_info
	{
.b32 1832
.b8 2
.b8 0
.b32 .debug_abbrev
.b8 8
.b8 1
.b8 108,103,101,110,102,101,58,32,69,68,71,32,54,46,54
.b8 0
.b8 4
.b8 0
.b8 47,104,111,109,101,47,103,105,110,47,68,101,115,107,116,111,112,47,83,97,102,101,67,85,68,65,47,101,120,97,109,112,108,101,115,47,115,97,102,101
.b8 99,97,99,104,101,46,99,117
.b8 0
.b32 .debug_line
.b8 47,104,111,109,101,47,103,105,110,47,68,101,115,107,116,111,112,47,83,97,102,101,67,85,68,65
.b8 0
.b64 0
.b8 2
.b8 70,82,69,69,68,95,77,69,77,95,68,69,86
.b8 0
.b32 165
.b8 1
.b8 1
.b8 60
.b8 5
.b8 9
.b8 3
.b64 FREED_MEM_DEV
.b8 70,82,69,69,68,95,77,69,77,95,68,69,86
.b8 0
.b8 3
.b8 98,111,111,108
.b8 0
.b8 2
.b8 1
.b8 2
.b8 100,95,116,97,98,108,101
.b8 0
.b32 208
.b8 1
.b8 5
.b8 25
.b8 4
.b8 9
.b8 3
.b64 d_table
.b8 100,95,116,97,98,108,101
.b8 0
.b8 4
.b32 217
.b32 12
.b8 5
.b8 95,90,78,56,115,97,102,101,99,117,100,97,54,109,101,109,111,114,121,49,53,65,108,108,111,99,97,116,105,111,110,84,97,98,108,101,69
.b8 0
.b8 16
.b8 1
.b8 43
.b8 6
.b8 101,110,116,114,105,101,115
.b8 0
.b32 313
.b8 1
.b8 44
.b8 2
.b8 35
.b8 0
.b8 6
.b8 99,111,117,110,116
.b8 0
.b32 463
.b8 1
.b8 45
.b8 2
.b8 35
.b8 8
.b8 6
.b8 99,97,112,97,99,105,116,121
.b8 0
.b32 463
.b8 1
.b8 46
.b8 2
.b8 35
.b8 12
.b8 0
.b8 4
.b32 322
.b32 12
.b8 5
.b8 95,90,78,56,115,97,102,101,99,117,100,97,54,109,101,109,111,114,121,53,69,110,116,114,121,69
.b8 0
.b8 24
.b8 1
.b8 30
.b8 6
.b8 115,116,97,114,116,95,97,100,100,114
.b8 0
.b32 429
.b8 1
.b8 31
.b8 2
.b8 35
.b8 0
.b8 6
.b8 98,108,111,99,107,95,115,105,122,101
.b8 0
.b32 463
.b8 1
.b8 32
.b8 2
.b8 35
.b8 8
.b8 6
.b8 102,108,97,103,115
.b8 0
.b32 463
.b8 1
.b8 33
.b8 2
.b8 35
.b8 12
.b8 6
.b8 101,112,111,99,104,115
.b8 0
.b32 463
.b8 1
.b8 34
.b8 2
.b8 35
.b8 16
.b8 0
.b8 7
.b32 446
.b8 117,105,110,116,112,116,114,95,116
.b8 0
.b8 2
.b8 79
.b8 3
.b8 117,110,115,105,103,110,101,100,32,108,111,110,103
.b8 0
.b8 7
.b8 8
.b8 7
.b32 479
.b8 117,105,110,116,51,50,95,116
.b8 0
.b8 4
.b8 26
.b8 7
.b32 497
.b8 95,95,117,105,110,116,51,50,95,116
.b8 0
.b8 3
.b8 42
.b8 3
.b8 117,110,115,105,103,110,101,100,32,105,110,116
.b8 0
.b8 7
.b8 4
.b8 7
.b32 531
.b8 95,95,117,105,110,116,49,54,95,116
.b8 0
.b8 3
.b8 40
.b8 3
.b8 117,110,115,105,103,110,101,100,32,115,104,111,114,116
.b8 0
.b8 7
.b8 2
.b8 7
.b32 513
.b8 117,105,110,116,49,54,95,116
.b8 0
.b8 4
.b8 25
.b8 7
.b32 582
.b8 95,95,117,105,110,116,56,95,116
.b8 0
.b8 3
.b8 38
.b8 3
.b8 117,110,115,105,103,110,101,100,32,99,104,97,114
.b8 0
.b8 8
.b8 1
.b8 7
.b32 565
.b8 117,105,110,116,56,95,116
.b8 0
.b8 4
.b8 24
.b8 5
.b8 95,90,78,56,115,97,102,101,99,117,100,97,54,109,101,109,111,114,121,56,77,101,116,97,100,97,116,97,69
.b8 0
.b8 16
.b8 1
.b8 37
.b8 6
.b8 109,97,103,105,99
.b8 0
.b32 549
.b8 1
.b8 38
.b8 2
.b8 35
.b8 0
.b8 6
.b8 112,97,100,100,105,110,103
.b8 0
.b32 699
.b8 1
.b8 39
.b8 2
.b8 35
.b8 2
.b8 6
.b8 101,110,116,114,121
.b8 0
.b32 313
.b8 1
.b8 40
.b8 2
.b8 35
.b8 8
.b8 0
.b8 8
.b32 599
.b8 9
.b32 711
.b8 6
.b8 0
.b8 10
.b8 95,95,65,82,82,65,89,95,83,73,90,69,95,84,89,80,69,95,95
.b8 0
.b8 8
.b8 7
.b8 7
.b32 751
.b8 95,95,105,110,116,51,50,95,116
.b8 0
.b8 3
.b8 41
.b8 3
.b8 105,110,116
.b8 0
.b8 5
.b8 4
.b8 7
.b32 734
.b8 105,110,116,51,50,95,116
.b8 0
.b8 6
.b8 26
.b8 11
.b64 $L__func_begin0
.b64 $L__func_end0
.b8 1
.b8 156
.b8 95,90,78,52,51,95,73,78,84,69,82,78,65,76,95,98,49,102,53,55,98,97,54,95,49,50,95,115,97,102,101,99,97,99,104,101,95,99,117,95
.b8 50,51,55,57,54,48,98,56,56,97,116,111,109,105,99,79,114,69,80,106,106
.b8 0
.b8 97,116,111,109,105,99,79,114
.b8 0
.b8 7
.b8 185
.b32 497
.b8 12
.b8 6
.b8 144
.b8 177
.b8 200
.b8 201
.b8 171
.b8 2
.b8 2
.b8 97,100,100,114,101,115,115
.b8 0
.b8 7
.b8 185
.b32 1793
.b8 12
.b8 5
.b8 144
.b8 177
.b8 228
.b8 149
.b8 1
.b8 2
.b8 118,97,108
.b8 0
.b8 7
.b8 185
.b32 497
.b8 0
.b8 13
.b64 $L__func_begin1
.b64 $L__func_end1
.b8 1
.b8 156
.b8 95,95,98,111,117,110,100,115,95,99,104,101,99,107,95,115,97,102,101,99,117,100,97
.b8 0
.b8 95,95,98,111,117,110,100,115,95,99,104,101,99,107,95,115,97,102,101,99,117,100,97
.b8 0
.b8 5
.b8 34
.b32 1787
.b8 1
.b8 12
.b8 6
.b8 144
.b8 181
.b8 200
.b8 201
.b8 171
.b8 2
.b8 2
.b8 112,116,114
.b8 0
.b8 5
.b8 34
.b32 1802
.b8 14
.b64 $L__tmp2
.b64 $L__tmp59
.b8 15
.b8 6
.b8 144
.b8 177
.b8 200
.b8 201
.b8 171
.b8 2
.b8 2
.b8 109,101,116,97
.b8 0
.b8 5
.b8 37
.b32 1811
.b8 15
.b8 6
.b8 144
.b8 181
.b8 200
.b8 201
.b8 171
.b8 2
.b8 2
.b8 97,100,100,114
.b8 0
.b8 5
.b8 76
.b32 1825
.b8 16
.b32 .debug_loc
.b8 102,114,101,101,100
.b8 0
.b8 5
.b8 77
.b32 165
.b8 16
.b32 .debug_loc+286
.b8 105,100,120
.b8 0
.b8 5
.b8 78
.b32 758
.b8 15
.b8 6
.b8 144
.b8 177
.b8 228
.b8 200
.b8 171
.b8 2
.b8 2
.b8 111,108,100
.b8 0
.b8 5
.b8 108
.b32 1830
.b8 14
.b64 $L__tmp4
.b64 $L__tmp17
.b8 17
.b8 6
.b8 11
.b8 3
.b64 __local_depot1
.b8 35
.b8 0
.b8 101,110,116,114,121
.b8 0
.b8 5
.b8 41
.b32 313
.b8 15
.b8 6
.b8 144
.b8 181
.b8 200
.b8 201
.b8 171
.b8 2
.b8 2
.b8 97,100,100,114
.b8 0
.b8 5
.b8 45
.b32 1825
.b8 15
.b8 6
.b8 144
.b8 178
.b8 226
.b8 200
.b8 171
.b8 2
.b8 2
.b8 111,108,100
.b8 0
.b8 5
.b8 65
.b32 1830
.b8 14
.b64 $L__tmp8
.b64 $L__tmp12
.b8 15
.b8 6
.b8 144
.b8 182
.b8 226
.b8 200
.b8 171
.b8 2
.b8 2
.b8 111,108,100
.b8 0
.b8 5
.b8 51
.b32 1830
.b8 0
.b8 0
.b8 14
.b64 $L__tmp20
.b64 $L__tmp50
.b8 16
.b32 .debug_loc+561
.b8 105
.b8 0
.b8 5
.b8 80
.b32 463
.b8 14
.b64 $L__tmp27
.b64 $L__tmp44
.b8 15
.b8 6
.b8 144
.b8 180
.b8 200
.b8 201
.b8 171
.b8 2
.b8 2
.b8 101,110,116,114,121
.b8 0
.b8 5
.b8 81
.b32 313
.b8 0
.b8 0
.b8 14
.b64 $L__tmp51
.b64 $L__tmp55
.b8 15
.b8 6
.b8 144
.b8 179
.b8 228
.b8 200
.b8 171
.b8 2
.b8 2
.b8 111,108,100
.b8 0
.b8 5
.b8 98
.b32 1830
.b8 0
.b8 0
.b8 0
.b8 13
.b64 $L__func_begin2
.b64 $L__func_end2
.b8 1
.b8 156
.b8 95,95,98,111,117,110,100,115,95,99,104,101,99,107,95,115,97,102,101,99,117,100,97,95,110,111,95,116,114,97,112
.b8 0
.b8 95,95,98,111,117,110,100,115,95,99,104,101,99,107,95,115,97,102,101,99,117,100,97,95,110,111,95,116,114,97,112
.b8 0
.b8 5
.b8 119
.b32 1787
.b8 1
.b8 12
.b8 6
.b8 144
.b8 181
.b8 200
.b8 201
.b8 171
.b8 2
.b8 2
.b8 112,116,114
.b8 0
.b8 5
.b8 119
.b32 1802
.b8 14
.b64 $L__tmp60
.b64 $L__tmp116
.b8 15
.b8 6
.b8 144
.b8 177
.b8 200
.b8 201
.b8 171
.b8 2
.b8 2
.b8 109,101,116,97
.b8 0
.b8 5
.b8 122
.b32 1811
.b8 15
.b8 6
.b8 144
.b8 181
.b8 200
.b8 201
.b8 171
.b8 2
.b8 2
.b8 97,100,100,114
.b8 0
.b8 5
.b8 161
.b32 1825
.b8 16
.b32 .debug_loc+694
.b8 102,114,101,101,100
.b8 0
.b8 5
.b8 162
.b32 165
.b8 16
.b32 .debug_loc+980
.b8 105,100,120
.b8 0
.b8 5
.b8 163
.b32 758
.b8 15
.b8 6
.b8 144
.b8 177
.b8 228
.b8 200
.b8 171
.b8 2
.b8 2
.b8 111,108,100
.b8 0
.b8 5
.b8 193
.b32 1830
.b8 14
.b64 $L__tmp62
.b64 $L__tmp74
.b8 17
.b8 6
.b8 11
.b8 3
.b64 __local_depot2
.b8 35
.b8 0
.b8 101,110,116,114,121
.b8 0
.b8 5
.b8 126
.b32 313
.b8 15
.b8 6
.b8 144
.b8 181
.b8 200
.b8 201
.b8 171
.b8 2
.b8 2
.b8 97,100,100,114
.b8 0
.b8 5
.b8 131
.b32 1825
.b8 15
.b8 6
.b8 144
.b8 178
.b8 226
.b8 200
.b8 171
.b8 2
.b8 2
.b8 111,108,100
.b8 0
.b8 5
.b8 151
.b32 1830
.b8 14
.b64 $L__tmp66
.b64 $L__tmp69
.b8 15
.b8 6
.b8 144
.b8 182
.b8 226
.b8 200
.b8 171
.b8 2
.b8 2
.b8 111,108,100
.b8 0
.b8 5
.b8 138
.b32 1830
.b8 0
.b8 0
.b8 14
.b64 $L__tmp77
.b64 $L__tmp107
.b8 16
.b32 .debug_loc+1255
.b8 105
.b8 0
.b8 5
.b8 164
.b32 463
.b8 14
.b64 $L__tmp84
.b64 $L__tmp101
.b8 15
.b8 6
.b8 144
.b8 180
.b8 200
.b8 201
.b8 171
.b8 2
.b8 2
.b8 101,110,116,114,121
.b8 0
.b8 5
.b8 165
.b32 313
.b8 0
.b8 0
.b8 14
.b64 $L__tmp108
.b64 $L__tmp112
.b8 15
.b8 6
.b8 144
.b8 179
.b8 228
.b8 200
.b8 171
.b8 2
.b8 2
.b8 111,108,100
.b8 0
.b8 5
.b8 183
.b32 1830
.b8 0
.b8 0
.b8 0
.b8 18
.b8 118,111,105,100
.b8 0
.b8 4
.b32 497
.b32 12
.b8 4
.b32 1787
.b32 12
.b8 4
.b32 1820
.b32 12
.b8 19
.b32 614
.b8 19
.b32 429
.b8 19
.b32 497
.b8 0
	}
	.section	.debug_macinfo
	{
.b8 0

	}


)ptx";

std::string no_trap_ver = R"ptx(
.extern .func  (.param .b32 func_retval0) vprintf
(
	.param .b64 vprintf_param_0,
	.param .b64 vprintf_param_1
)
;
.func __trap
()
;
.func  (.param .b32 func_retval0) __uAtomicOr
(
	.param .b64 __uAtomicOr_param_0,
	.param .b32 __uAtomicOr_param_1
)
;
.global .align 1 .u8 FREED_MEM_DEV;
.const .align 8 .u64 d_table;
.global .align 1 .b8 $str[6] = {72, 101, 114, 101, 10};

.func  (.param .b32 func_retval0) _ZN43_INTERNAL_b1f57ba6_12_safecache_cu_237960b88atomicOrEPjj(
	.param .b64 _ZN43_INTERNAL_b1f57ba6_12_safecache_cu_237960b88atomicOrEPjj_param_0,
	.param .b32 _ZN43_INTERNAL_b1f57ba6_12_safecache_cu_237960b88atomicOrEPjj_param_1
)
{
	.reg .b32 	%r<4>;
	.reg .b64 	%rd<3>;
	.loc	7 185 0
$L__func_begin0:
	.loc	7 185 0


	ld.param.u64 	%rd1, [_ZN43_INTERNAL_b1f57ba6_12_safecache_cu_237960b88atomicOrEPjj_param_0];
	ld.param.u32 	%r1, [_ZN43_INTERNAL_b1f57ba6_12_safecache_cu_237960b88atomicOrEPjj_param_1];
$L__tmp0:
	.loc	7 187 3
	mov.b64 	%rd2, %rd1;
	mov.b32 	%r2, %r1;
	{ // callseq 0, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd2;
	.param .b32 param1;
	st.param.b32 	[param1+0], %r2;
	.param .b32 retval0;
	call.uni (retval0),
	__uAtomicOr,
	(
	param0,
	param1
	);
	ld.param.b32 	%r3, [retval0+0];
	} // callseq 0
	st.param.b32 	[func_retval0+0], %r3;
	ret;
$L__tmp1:
$L__func_end0:

}

.func __bounds_check_safecuda_no_trap(
	.param .b64 __bounds_check_safecuda_no_trap_param_0
)
{
	.local .align 8 .b8 	__local_depot2[8];
	.reg .b64 	%SP;
	.reg .b64 	%SPL;
	.reg .pred 	%p<37>;
	.reg .b16 	%rs<13>;
	.reg .b32 	%r<33>;
	.reg .b64 	%rd<70>;
	.loc	5 119 0
$L__func_begin2:
	.loc	5 119 0


	mov.u64 	%SPL, __local_depot2;
	cvta.local.u64 	%SP, %SPL;
	ld.param.u64 	%rd5, [__bounds_check_safecuda_no_trap_param_0];
$L__tmp60:
	.loc	5 121 2
	mov.u64 	%rd6, $str;
	cvta.global.u64 	%rd7, %rd6;
	{ // callseq 10, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd7;
	.param .b64 param1;
	st.param.b64 	[param1+0], 0;
	.param .b32 retval0;
	call.uni (retval0),
	vprintf,
	(
	param0,
	param1
	);
	ld.param.b32 	%r9, [retval0+0];
	} // callseq 10
	.loc	5 123 3
	add.s64 	%rd1, %rd5, -16;
$L__tmp61:
	.loc	5 125 2
	ld.u16 	%rs6, [%rd1];
	cvt.u32.u16 	%r10, %rs6;
	setp.eq.s32 	%p5, %r10, 23294;
	not.pred 	%p6, %p5;
	@%p6 bra 	$L__BB2_14;
	bra.uni 	$L__BB2_1;

$L__BB2_1:
$L__tmp62:
	.loc	5 126 3
	ld.u64 	%rd8, [%rd1+8];
	mov.b64 	%rd9, %rd8;
	st.u64 	[%SP+0], %rd9;
	.loc	5 128 3
	ld.u64 	%rd10, [%SP+0];
	setp.eq.s64 	%p7, %rd10, 0;
	not.pred 	%p8, %p7;
	@%p8 bra 	$L__BB2_3;
	bra.uni 	$L__BB2_2;

$L__BB2_2:
$L__tmp63:
	.loc	5 129 4
	bra.uni 	$L__BB2_15;
$L__tmp64:

$L__BB2_3:
	.loc	5 133 3
	ld.u64 	%rd11, [%SP+0];
	ld.u64 	%rd12, [%rd11];
	setp.ge.u64 	%p10, %rd5, %rd12;
	mov.pred 	%p9, 0;
	not.pred 	%p11, %p10;
	mov.pred 	%p35, %p9;
	@%p11 bra 	$L__BB2_5;
	bra.uni 	$L__BB2_4;

$L__BB2_4:
	.loc	5 134 7
	ld.u64 	%rd13, [%SP+0];
	ld.u64 	%rd14, [%rd13];
	ld.u64 	%rd15, [%SP+0];
	ld.u32 	%r11, [%rd15+8];
	cvt.u64.u32 	%rd16, %r11;
	add.s64 	%rd17, %rd14, %rd16;
	setp.lt.u64 	%p1, %rd5, %rd17;
	mov.pred 	%p35, %p1;
	bra.uni 	$L__BB2_5;

$L__BB2_5:
	mov.pred 	%p2, %p35;
	not.pred 	%p12, %p2;
	@%p12 bra 	$L__BB2_11;
	bra.uni 	$L__BB2_6;

$L__BB2_6:
$L__tmp65:
	.loc	5 136 4
	ld.u64 	%rd27, [%SP+0];
	ld.u32 	%r14, [%rd27+12];
	.loc	5 137 8
	and.b32  	%r15, %r14, 2;
	setp.ne.s32 	%p15, %r15, 0;
	not.pred 	%p16, %p15;
	@%p16 bra 	$L__BB2_10;
	bra.uni 	$L__BB2_7;

$L__BB2_7:
$L__tmp66:
	.loc	5 139 6
	mov.u64 	%rd28, d_table;
	cvta.const.u64 	%rd29, %rd28;
	ld.u64 	%rd30, [%rd29];
	ld.u64 	%rd31, [%rd30];
	add.s64 	%rd32, %rd31, 12;
	.loc	5 138 22
	{ // callseq 12, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd32;
	.param .b32 param1;
	st.param.b32 	[param1+0], 2;
	.param .b32 retval0;
	call.uni (retval0),
	_ZN43_INTERNAL_b1f57ba6_12_safecache_cu_237960b88atomicOrEPjj,
	(
	param0,
	param1
	);
	ld.param.b32 	%r16, [retval0+0];
$L__tmp67:
	} // callseq 12
	.loc	5 142 10
	and.b32  	%r17, %r16, 2;
	.loc	5 143 9
	setp.eq.s32 	%p17, %r17, 0;
	not.pred 	%p18, %p17;
	@%p18 bra 	$L__BB2_9;
	bra.uni 	$L__BB2_8;

$L__BB2_8:
$L__tmp68:
	.loc	5 144 6
	mov.u64 	%rd33, d_table;
	cvta.const.u64 	%rd34, %rd33;
	ld.u64 	%rd35, [%rd34];
	ld.u64 	%rd36, [%rd35];
	st.u64 	[%rd36], %rd5;
	bra.uni 	$L__BB2_9;

$L__BB2_9:
	bra.uni 	$L__BB2_10;
$L__tmp69:

$L__BB2_10:
	.loc	5 148 4
	bra.uni 	$L__BB2_34;
$L__tmp70:

$L__BB2_11:
	.loc	5 152 4
	mov.u64 	%rd18, d_table;
	cvta.const.u64 	%rd19, %rd18;
	ld.u64 	%rd20, [%rd19];
	ld.u64 	%rd21, [%rd20];
	add.s64 	%rd22, %rd21, 12;
	{ // callseq 11, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd22;
	.param .b32 param1;
	st.param.b32 	[param1+0], 1;
	.param .b32 retval0;
	call.uni (retval0),
	_ZN43_INTERNAL_b1f57ba6_12_safecache_cu_237960b88atomicOrEPjj,
	(
	param0,
	param1
	);
	ld.param.b32 	%r12, [retval0+0];
$L__tmp71:
	} // callseq 11
	.loc	5 154 3
	and.b32  	%r13, %r12, 1;
	setp.eq.s32 	%p13, %r13, 0;
	not.pred 	%p14, %p13;
	@%p14 bra 	$L__BB2_13;
	bra.uni 	$L__BB2_12;

$L__BB2_12:
$L__tmp72:
	.loc	5 155 4
	mov.u64 	%rd23, d_table;
	cvta.const.u64 	%rd24, %rd23;
	ld.u64 	%rd25, [%rd24];
	ld.u64 	%rd26, [%rd25];
	st.u64 	[%rd26], %rd5;
	bra.uni 	$L__BB2_13;
$L__tmp73:

$L__BB2_13:
	.loc	5 158 3
	bra.uni 	$L__BB2_34;
$L__tmp74:

$L__BB2_14:
	.loc	5 160 1
	bra.uni 	$L__BB2_15;

$L__BB2_15:
$L__tmp75:
	.loc	5 162 2
	mov.u16 	%rs7, 0;
	mov.b16 	%rs1, %rs7;
$L__tmp76:
	.loc	5 163 2
	mov.u32 	%r18, -1;
	mov.b32 	%r1, %r18;
$L__tmp77:
	.loc	5 164 2
	mov.u32 	%r19, 1;
	mov.b32 	%r2, %r19;
$L__tmp78:
	mov.u16 	%rs10, %rs1;
$L__tmp79:
	mov.u32 	%r29, %r1;
$L__tmp80:
	mov.u32 	%r30, %r2;
$L__tmp81:
	bra.uni 	$L__BB2_16;

$L__BB2_16:
	mov.u32 	%r4, %r30;
	mov.u32 	%r3, %r29;
	mov.u16 	%rs2, %rs10;
$L__tmp82:
	mov.u64 	%rd37, d_table;
$L__tmp83:
	cvta.const.u64 	%rd38, %rd37;
	ld.u64 	%rd39, [%rd38];
	ld.u32 	%r20, [%rd39+8];
	setp.lt.u32 	%p19, %r4, %r20;
	not.pred 	%p20, %p19;
	@%p20 bra 	$L__BB2_27;
	bra.uni 	$L__BB2_17;

$L__BB2_17:
$L__tmp84:
	.loc	5 165 3
	mov.u64 	%rd60, d_table;
	cvta.const.u64 	%rd61, %rd60;
	ld.u64 	%rd62, [%rd61];
	ld.u64 	%rd63, [%rd62];
	cvt.u64.u32 	%rd64, %r4;
	mul.lo.s64 	%rd65, %rd64, 24;
	add.s64 	%rd4, %rd63, %rd65;
$L__tmp85:
	.loc	5 167 3
	ld.u64 	%rd66, [%rd4];
	setp.le.u64 	%p28, %rd66, %rd5;
	mov.pred 	%p27, 0;
	not.pred 	%p29, %p28;
	mov.pred 	%p36, %p27;
	@%p29 bra 	$L__BB2_19;
	bra.uni 	$L__BB2_18;

$L__BB2_18:
	.loc	5 168 7
	ld.u64 	%rd67, [%rd4];
	ld.u32 	%r25, [%rd4+8];
	cvt.u64.u32 	%rd68, %r25;
	add.s64 	%rd69, %rd67, %rd68;
	setp.lt.u64 	%p3, %rd5, %rd69;
	mov.pred 	%p36, %p3;
	bra.uni 	$L__BB2_19;

$L__BB2_19:
	mov.pred 	%p4, %p36;
	not.pred 	%p30, %p4;
	mov.u16 	%rs12, %rs2;
$L__tmp86:
	mov.u32 	%r32, %r3;
$L__tmp87:
	@%p30 bra 	$L__BB2_25;
	bra.uni 	$L__BB2_20;

$L__BB2_20:
$L__tmp88:
	.loc	5 170 4
	ld.u32 	%r26, [%rd4+12];
	setp.eq.s32 	%p31, %r26, 0;
	not.pred 	%p32, %p31;
	@%p32 bra 	$L__BB2_22;
	bra.uni 	$L__BB2_21;

$L__BB2_21:
$L__tmp89:
	.loc	5 171 5
	bra.uni 	$L__BB2_34;
$L__tmp90:

$L__BB2_22:
	.loc	5 174 4
	ld.u32 	%r27, [%rd4+12];
	.loc	5 175 8
	and.b32  	%r28, %r27, 2;
	setp.ne.s32 	%p33, %r28, 0;
	not.pred 	%p34, %p33;
	mov.u16 	%rs11, %rs2;
$L__tmp91:
	mov.u32 	%r31, %r3;
$L__tmp92:
	@%p34 bra 	$L__BB2_24;
	bra.uni 	$L__BB2_23;

$L__BB2_23:
$L__tmp93:
	.loc	5 176 5
	mov.u16 	%rs9, 1;
	mov.b16 	%rs3, %rs9;
$L__tmp94:
	.loc	5 177 5
	mov.b32 	%r5, %r4;
$L__tmp95:
	mov.u16 	%rs11, %rs3;
$L__tmp96:
	mov.u32 	%r31, %r5;
$L__tmp97:
	bra.uni 	$L__BB2_24;

$L__BB2_24:
	mov.u32 	%r6, %r31;
	mov.u16 	%rs4, %rs11;
$L__tmp98:
	mov.u16 	%rs12, %rs4;
$L__tmp99:
	mov.u32 	%r32, %r6;
$L__tmp100:
	bra.uni 	$L__BB2_25;
$L__tmp101:

$L__BB2_25:
	.loc	5 164 48
	mov.u32 	%r7, %r32;
	mov.u16 	%rs5, %rs12;
$L__tmp102:
	bra.uni 	$L__BB2_26;

$L__BB2_26:
	add.s32 	%r8, %r4, 1;
$L__tmp103:
	mov.u16 	%rs10, %rs5;
$L__tmp104:
	mov.u32 	%r29, %r7;
$L__tmp105:
	mov.u32 	%r30, %r8;
$L__tmp106:
	bra.uni 	$L__BB2_16;
$L__tmp107:

$L__BB2_27:
	.loc	5 182 2
	and.b16  	%rs8, %rs2, 255;
	setp.ne.s16 	%p21, %rs8, 0;
	not.pred 	%p22, %p21;
	@%p22 bra 	$L__BB2_31;
	bra.uni 	$L__BB2_28;

$L__BB2_28:
$L__tmp108:
	.loc	5 183 3
	mov.u64 	%rd49, d_table;
	cvta.const.u64 	%rd50, %rd49;
	ld.u64 	%rd51, [%rd50];
	ld.u64 	%rd52, [%rd51];
	add.s64 	%rd53, %rd52, 12;
	.loc	5 183 20
	{ // callseq 14, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd53;
	.param .b32 param1;
	st.param.b32 	[param1+0], 2;
	.param .b32 retval0;
	call.uni (retval0),
	_ZN43_INTERNAL_b1f57ba6_12_safecache_cu_237960b88atomicOrEPjj,
	(
	param0,
	param1
	);
	ld.param.b32 	%r23, [retval0+0];
$L__tmp109:
	} // callseq 14
	.loc	5 185 3
	and.b32  	%r24, %r23, 2;
	setp.eq.s32 	%p25, %r24, 0;
	not.pred 	%p26, %p25;
	@%p26 bra 	$L__BB2_30;
	bra.uni 	$L__BB2_29;

$L__BB2_29:
$L__tmp110:
	.loc	5 186 4
	mov.u64 	%rd54, d_table;
	cvta.const.u64 	%rd55, %rd54;
	ld.u64 	%rd56, [%rd55];
	ld.u64 	%rd57, [%rd56];
	st.u64 	[%rd57], %rd5;
	.loc	5 187 4
	ld.u64 	%rd58, [%rd55];
	ld.u64 	%rd59, [%rd58];
	st.u32 	[%rd59+8], %r3;
	bra.uni 	$L__BB2_30;
$L__tmp111:

$L__BB2_30:
	.loc	5 190 3
	bra.uni 	$L__BB2_34;
$L__tmp112:

$L__BB2_31:
	.loc	5 193 2
	mov.u64 	%rd40, d_table;
	cvta.const.u64 	%rd41, %rd40;
	ld.u64 	%rd42, [%rd41];
	ld.u64 	%rd43, [%rd42];
	add.s64 	%rd44, %rd43, 12;
	.loc	5 193 19
	{ // callseq 13, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd44;
	.param .b32 param1;
	st.param.b32 	[param1+0], 1;
	.param .b32 retval0;
	call.uni (retval0),
	_ZN43_INTERNAL_b1f57ba6_12_safecache_cu_237960b88atomicOrEPjj,
	(
	param0,
	param1
	);
	ld.param.b32 	%r21, [retval0+0];
$L__tmp113:
	} // callseq 13
	.loc	5 196 2
	and.b32  	%r22, %r21, 1;
	setp.eq.s32 	%p23, %r22, 0;
	not.pred 	%p24, %p23;
	@%p24 bra 	$L__BB2_33;
	bra.uni 	$L__BB2_32;

$L__BB2_32:
$L__tmp114:
	.loc	5 197 3
	mov.u64 	%rd45, d_table;
	cvta.const.u64 	%rd46, %rd45;
	ld.u64 	%rd47, [%rd46];
	ld.u64 	%rd48, [%rd47];
	st.u64 	[%rd48], %rd5;
	bra.uni 	$L__BB2_33;
$L__tmp115:

$L__BB2_33:
	.loc	5 200 1
	bra.uni 	$L__BB2_34;

$L__BB2_34:
	ret;
$L__tmp116:
$L__func_end2:

}
.func __trap()
{



	// begin inline asm
	trap;
	// end inline asm
	ret;
$L__func_end3:

}
.func  (.param .b32 func_retval0) __uAtomicOr(
	.param .b64 __uAtomicOr_param_0,
	.param .b32 __uAtomicOr_param_1
)
{
	.reg .b32 	%r<3>;
	.reg .b64 	%rd<2>;


	ld.param.u64 	%rd1, [__uAtomicOr_param_0];
	ld.param.u32 	%r1, [__uAtomicOr_param_1];
	atom.or.b32 	%r2, [%rd1], %r1;
	st.param.b32 	[func_retval0+0], %r2;
	ret;
$L__func_end4:

}
	.file	1 "/home/gin/Desktop/SafeCUDA/examples/safecache.cuh"
	.file	2 "/usr/include/stdint.h"
	.file	3 "/usr/include/bits/types.h"
	.file	4 "/usr/include/bits/stdint-uintn.h"
	.file	5 "/home/gin/Desktop/SafeCUDA/examples/safecache.cu"
	.file	6 "/usr/include/bits/stdint-intn.h"
	.file	7 "/usr/local/cuda/bin/../targets/x86_64-linux/include/device_atomic_functions.hpp"
	.section	.debug_loc
	{
.b64 $L__tmp19
.b64 $L__tmp22
.b8 6
.b8 0
.b8 144
.b8 177
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp22
.b64 $L__tmp26
.b8 7
.b8 0
.b8 144
.b8 178
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp26
.b64 $L__tmp29
.b8 6
.b8 0
.b8 144
.b8 178
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp29
.b64 $L__tmp34
.b8 7
.b8 0
.b8 144
.b8 180
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp34
.b64 $L__tmp37
.b8 7
.b8 0
.b8 144
.b8 179
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp37
.b64 $L__tmp39
.b8 6
.b8 0
.b8 144
.b8 179
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp39
.b64 $L__tmp41
.b8 7
.b8 0
.b8 144
.b8 179
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp41
.b64 $L__tmp42
.b8 6
.b8 0
.b8 144
.b8 180
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp42
.b64 $L__tmp45
.b8 7
.b8 0
.b8 144
.b8 180
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp45
.b64 $L__tmp47
.b8 6
.b8 0
.b8 144
.b8 181
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp47
.b64 $L__func_end1
.b8 7
.b8 0
.b8 144
.b8 178
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 0
.b64 0
.b64 $L__tmp20
.b64 $L__tmp23
.b8 5
.b8 0
.b8 144
.b8 177
.b8 228
.b8 149
.b8 1
.b64 $L__tmp23
.b64 $L__tmp25
.b8 6
.b8 0
.b8 144
.b8 185
.b8 228
.b8 200
.b8 171
.b8 2
.b64 $L__tmp25
.b64 $L__tmp30
.b8 5
.b8 0
.b8 144
.b8 179
.b8 228
.b8 149
.b8 1
.b64 $L__tmp30
.b64 $L__tmp35
.b8 6
.b8 0
.b8 144
.b8 178
.b8 230
.b8 200
.b8 171
.b8 2
.b64 $L__tmp35
.b64 $L__tmp38
.b8 6
.b8 0
.b8 144
.b8 177
.b8 230
.b8 200
.b8 171
.b8 2
.b64 $L__tmp38
.b64 $L__tmp40
.b8 5
.b8 0
.b8 144
.b8 181
.b8 228
.b8 149
.b8 1
.b64 $L__tmp40
.b64 $L__tmp41
.b8 6
.b8 0
.b8 144
.b8 177
.b8 230
.b8 200
.b8 171
.b8 2
.b64 $L__tmp41
.b64 $L__tmp43
.b8 5
.b8 0
.b8 144
.b8 182
.b8 228
.b8 149
.b8 1
.b64 $L__tmp43
.b64 $L__tmp45
.b8 6
.b8 0
.b8 144
.b8 178
.b8 230
.b8 200
.b8 171
.b8 2
.b64 $L__tmp45
.b64 $L__tmp48
.b8 5
.b8 0
.b8 144
.b8 183
.b8 228
.b8 149
.b8 1
.b64 $L__tmp48
.b64 $L__func_end1
.b8 6
.b8 0
.b8 144
.b8 185
.b8 228
.b8 200
.b8 171
.b8 2
.b64 0
.b64 0
.b64 $L__tmp21
.b64 $L__tmp24
.b8 5
.b8 0
.b8 144
.b8 178
.b8 228
.b8 149
.b8 1
.b64 $L__tmp24
.b64 $L__tmp26
.b8 6
.b8 0
.b8 144
.b8 176
.b8 230
.b8 200
.b8 171
.b8 2
.b64 $L__tmp26
.b64 $L__tmp46
.b8 5
.b8 0
.b8 144
.b8 180
.b8 228
.b8 149
.b8 1
.b64 $L__tmp46
.b64 $L__tmp49
.b8 5
.b8 0
.b8 144
.b8 184
.b8 228
.b8 149
.b8 1
.b64 $L__tmp49
.b64 $L__func_end1
.b8 6
.b8 0
.b8 144
.b8 176
.b8 230
.b8 200
.b8 171
.b8 2
.b64 0
.b64 0
.b64 $L__tmp76
.b64 $L__tmp79
.b8 6
.b8 0
.b8 144
.b8 177
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp79
.b64 $L__tmp83
.b8 7
.b8 0
.b8 144
.b8 176
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp83
.b64 $L__tmp86
.b8 6
.b8 0
.b8 144
.b8 178
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp86
.b64 $L__tmp91
.b8 7
.b8 0
.b8 144
.b8 178
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp91
.b64 $L__tmp94
.b8 7
.b8 0
.b8 144
.b8 177
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp94
.b64 $L__tmp96
.b8 6
.b8 0
.b8 144
.b8 179
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp96
.b64 $L__tmp98
.b8 7
.b8 0
.b8 144
.b8 177
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp98
.b64 $L__tmp99
.b8 6
.b8 0
.b8 144
.b8 180
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp99
.b64 $L__tmp102
.b8 7
.b8 0
.b8 144
.b8 178
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp102
.b64 $L__tmp104
.b8 6
.b8 0
.b8 144
.b8 181
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp104
.b64 $L__func_end2
.b8 7
.b8 0
.b8 144
.b8 176
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 0
.b64 0
.b64 $L__tmp77
.b64 $L__tmp80
.b8 5
.b8 0
.b8 144
.b8 177
.b8 228
.b8 149
.b8 1
.b64 $L__tmp80
.b64 $L__tmp82
.b8 6
.b8 0
.b8 144
.b8 185
.b8 228
.b8 200
.b8 171
.b8 2
.b64 $L__tmp82
.b64 $L__tmp87
.b8 5
.b8 0
.b8 144
.b8 179
.b8 228
.b8 149
.b8 1
.b64 $L__tmp87
.b64 $L__tmp92
.b8 6
.b8 0
.b8 144
.b8 178
.b8 230
.b8 200
.b8 171
.b8 2
.b64 $L__tmp92
.b64 $L__tmp95
.b8 6
.b8 0
.b8 144
.b8 177
.b8 230
.b8 200
.b8 171
.b8 2
.b64 $L__tmp95
.b64 $L__tmp97
.b8 5
.b8 0
.b8 144
.b8 181
.b8 228
.b8 149
.b8 1
.b64 $L__tmp97
.b64 $L__tmp98
.b8 6
.b8 0
.b8 144
.b8 177
.b8 230
.b8 200
.b8 171
.b8 2
.b64 $L__tmp98
.b64 $L__tmp100
.b8 5
.b8 0
.b8 144
.b8 182
.b8 228
.b8 149
.b8 1
.b64 $L__tmp100
.b64 $L__tmp102
.b8 6
.b8 0
.b8 144
.b8 178
.b8 230
.b8 200
.b8 171
.b8 2
.b64 $L__tmp102
.b64 $L__tmp105
.b8 5
.b8 0
.b8 144
.b8 183
.b8 228
.b8 149
.b8 1
.b64 $L__tmp105
.b64 $L__func_end2
.b8 6
.b8 0
.b8 144
.b8 185
.b8 228
.b8 200
.b8 171
.b8 2
.b64 0
.b64 0
.b64 $L__tmp78
.b64 $L__tmp81
.b8 5
.b8 0
.b8 144
.b8 178
.b8 228
.b8 149
.b8 1
.b64 $L__tmp81
.b64 $L__tmp83
.b8 6
.b8 0
.b8 144
.b8 176
.b8 230
.b8 200
.b8 171
.b8 2
.b64 $L__tmp83
.b64 $L__tmp103
.b8 5
.b8 0
.b8 144
.b8 180
.b8 228
.b8 149
.b8 1
.b64 $L__tmp103
.b64 $L__tmp106
.b8 5
.b8 0
.b8 144
.b8 184
.b8 228
.b8 149
.b8 1
.b64 $L__tmp106
.b64 $L__func_end2
.b8 6
.b8 0
.b8 144
.b8 176
.b8 230
.b8 200
.b8 171
.b8 2
.b64 0
.b64 0
	}
	.section	.debug_abbrev
	{
.b8 1
.b8 17
.b8 1
.b8 37
.b8 8
.b8 19
.b8 5
.b8 3
.b8 8
.b8 16
.b8 6
.b8 27
.b8 8
.b8 17
.b8 1
.b8 0
.b8 0
.b8 2
.b8 52
.b8 0
.b8 3
.b8 8
.b8 73
.b8 19
.b8 63
.b8 12
.b8 58
.b8 11
.b8 59
.b8 11
.b8 51
.b8 11
.b8 2
.b8 10
.b8 135,64
.b8 8
.b8 0
.b8 0
.b8 3
.b8 36
.b8 0
.b8 3
.b8 8
.b8 62
.b8 11
.b8 11
.b8 11
.b8 0
.b8 0
.b8 4
.b8 15
.b8 0
.b8 73
.b8 19
.b8 51
.b8 6
.b8 0
.b8 0
.b8 5
.b8 19
.b8 1
.b8 3
.b8 8
.b8 11
.b8 11
.b8 58
.b8 11
.b8 59
.b8 11
.b8 0
.b8 0
.b8 6
.b8 13
.b8 0
.b8 3
.b8 8
.b8 73
.b8 19
.b8 58
.b8 11
.b8 59
.b8 11
.b8 56
.b8 10
.b8 0
.b8 0
.b8 7
.b8 22
.b8 0
.b8 73
.b8 19
.b8 3
.b8 8
.b8 58
.b8 11
.b8 59
.b8 11
.b8 0
.b8 0
.b8 8
.b8 1
.b8 1
.b8 73
.b8 19
.b8 0
.b8 0
.b8 9
.b8 33
.b8 0
.b8 73
.b8 19
.b8 55
.b8 11
.b8 0
.b8 0
.b8 10
.b8 36
.b8 0
.b8 3
.b8 8
.b8 11
.b8 11
.b8 62
.b8 11
.b8 0
.b8 0
.b8 11
.b8 46
.b8 1
.b8 17
.b8 1
.b8 18
.b8 1
.b8 64
.b8 10
.b8 135,64
.b8 8
.b8 3
.b8 8
.b8 58
.b8 11
.b8 59
.b8 11
.b8 73
.b8 19
.b8 0
.b8 0
.b8 12
.b8 5
.b8 0
.b8 2
.b8 10
.b8 51
.b8 11
.b8 3
.b8 8
.b8 58
.b8 11
.b8 59
.b8 11
.b8 73
.b8 19
.b8 0
.b8 0
.b8 13
.b8 46
.b8 1
.b8 17
.b8 1
.b8 18
.b8 1
.b8 64
.b8 10
.b8 135,64
.b8 8
.b8 3
.b8 8
.b8 58
.b8 11
.b8 59
.b8 11
.b8 73
.b8 19
.b8 63
.b8 12
.b8 0
.b8 0
.b8 14
.b8 11
.b8 1
.b8 17
.b8 1
.b8 18
.b8 1
.b8 0
.b8 0
.b8 15
.b8 52
.b8 0
.b8 2
.b8 10
.b8 51
.b8 11
.b8 3
.b8 8
.b8 58
.b8 11
.b8 59
.b8 11
.b8 73
.b8 19
.b8 0
.b8 0
.b8 16
.b8 52
.b8 0
.b8 2
.b8 6
.b8 3
.b8 8
.b8 58
.b8 11
.b8 59
.b8 11
.b8 73
.b8 19
.b8 0
.b8 0
.b8 17
.b8 52
.b8 0
.b8 51
.b8 11
.b8 2
.b8 10
.b8 3
.b8 8
.b8 58
.b8 11
.b8 59
.b8 11
.b8 73
.b8 19
.b8 0
.b8 0
.b8 18
.b8 59
.b8 0
.b8 3
.b8 8
.b8 0
.b8 0
.b8 19
.b8 38
.b8 0
.b8 73
.b8 19
.b8 0
.b8 0
.b8 0
	}
	.section	.debug_info
	{
.b32 1832
.b8 2
.b8 0
.b32 .debug_abbrev
.b8 8
.b8 1
.b8 108,103,101,110,102,101,58,32,69,68,71,32,54,46,54
.b8 0
.b8 4
.b8 0
.b8 47,104,111,109,101,47,103,105,110,47,68,101,115,107,116,111,112,47,83,97,102,101,67,85,68,65,47,101,120,97,109,112,108,101,115,47,115,97,102,101
.b8 99,97,99,104,101,46,99,117
.b8 0
.b32 .debug_line
.b8 47,104,111,109,101,47,103,105,110,47,68,101,115,107,116,111,112,47,83,97,102,101,67,85,68,65
.b8 0
.b64 0
.b8 2
.b8 70,82,69,69,68,95,77,69,77,95,68,69,86
.b8 0
.b32 165
.b8 1
.b8 1
.b8 60
.b8 5
.b8 9
.b8 3
.b64 FREED_MEM_DEV
.b8 70,82,69,69,68,95,77,69,77,95,68,69,86
.b8 0
.b8 3
.b8 98,111,111,108
.b8 0
.b8 2
.b8 1
.b8 2
.b8 100,95,116,97,98,108,101
.b8 0
.b32 208
.b8 1
.b8 5
.b8 25
.b8 4
.b8 9
.b8 3
.b64 d_table
.b8 100,95,116,97,98,108,101
.b8 0
.b8 4
.b32 217
.b32 12
.b8 5
.b8 95,90,78,56,115,97,102,101,99,117,100,97,54,109,101,109,111,114,121,49,53,65,108,108,111,99,97,116,105,111,110,84,97,98,108,101,69
.b8 0
.b8 16
.b8 1
.b8 43
.b8 6
.b8 101,110,116,114,105,101,115
.b8 0
.b32 313
.b8 1
.b8 44
.b8 2
.b8 35
.b8 0
.b8 6
.b8 99,111,117,110,116
.b8 0
.b32 463
.b8 1
.b8 45
.b8 2
.b8 35
.b8 8
.b8 6
.b8 99,97,112,97,99,105,116,121
.b8 0
.b32 463
.b8 1
.b8 46
.b8 2
.b8 35
.b8 12
.b8 0
.b8 4
.b32 322
.b32 12
.b8 5
.b8 95,90,78,56,115,97,102,101,99,117,100,97,54,109,101,109,111,114,121,53,69,110,116,114,121,69
.b8 0
.b8 24
.b8 1
.b8 30
.b8 6
.b8 115,116,97,114,116,95,97,100,100,114
.b8 0
.b32 429
.b8 1
.b8 31
.b8 2
.b8 35
.b8 0
.b8 6
.b8 98,108,111,99,107,95,115,105,122,101
.b8 0
.b32 463
.b8 1
.b8 32
.b8 2
.b8 35
.b8 8
.b8 6
.b8 102,108,97,103,115
.b8 0
.b32 463
.b8 1
.b8 33
.b8 2
.b8 35
.b8 12
.b8 6
.b8 101,112,111,99,104,115
.b8 0
.b32 463
.b8 1
.b8 34
.b8 2
.b8 35
.b8 16
.b8 0
.b8 7
.b32 446
.b8 117,105,110,116,112,116,114,95,116
.b8 0
.b8 2
.b8 79
.b8 3
.b8 117,110,115,105,103,110,101,100,32,108,111,110,103
.b8 0
.b8 7
.b8 8
.b8 7
.b32 479
.b8 117,105,110,116,51,50,95,116
.b8 0
.b8 4
.b8 26
.b8 7
.b32 497
.b8 95,95,117,105,110,116,51,50,95,116
.b8 0
.b8 3
.b8 42
.b8 3
.b8 117,110,115,105,103,110,101,100,32,105,110,116
.b8 0
.b8 7
.b8 4
.b8 7
.b32 531
.b8 95,95,117,105,110,116,49,54,95,116
.b8 0
.b8 3
.b8 40
.b8 3
.b8 117,110,115,105,103,110,101,100,32,115,104,111,114,116
.b8 0
.b8 7
.b8 2
.b8 7
.b32 513
.b8 117,105,110,116,49,54,95,116
.b8 0
.b8 4
.b8 25
.b8 7
.b32 582
.b8 95,95,117,105,110,116,56,95,116
.b8 0
.b8 3
.b8 38
.b8 3
.b8 117,110,115,105,103,110,101,100,32,99,104,97,114
.b8 0
.b8 8
.b8 1
.b8 7
.b32 565
.b8 117,105,110,116,56,95,116
.b8 0
.b8 4
.b8 24
.b8 5
.b8 95,90,78,56,115,97,102,101,99,117,100,97,54,109,101,109,111,114,121,56,77,101,116,97,100,97,116,97,69
.b8 0
.b8 16
.b8 1
.b8 37
.b8 6
.b8 109,97,103,105,99
.b8 0
.b32 549
.b8 1
.b8 38
.b8 2
.b8 35
.b8 0
.b8 6
.b8 112,97,100,100,105,110,103
.b8 0
.b32 699
.b8 1
.b8 39
.b8 2
.b8 35
.b8 2
.b8 6
.b8 101,110,116,114,121
.b8 0
.b32 313
.b8 1
.b8 40
.b8 2
.b8 35
.b8 8
.b8 0
.b8 8
.b32 599
.b8 9
.b32 711
.b8 6
.b8 0
.b8 10
.b8 95,95,65,82,82,65,89,95,83,73,90,69,95,84,89,80,69,95,95
.b8 0
.b8 8
.b8 7
.b8 7
.b32 751
.b8 95,95,105,110,116,51,50,95,116
.b8 0
.b8 3
.b8 41
.b8 3
.b8 105,110,116
.b8 0
.b8 5
.b8 4
.b8 7
.b32 734
.b8 105,110,116,51,50,95,116
.b8 0
.b8 6
.b8 26
.b8 11
.b64 $L__func_begin0
.b64 $L__func_end0
.b8 1
.b8 156
.b8 95,90,78,52,51,95,73,78,84,69,82,78,65,76,95,98,49,102,53,55,98,97,54,95,49,50,95,115,97,102,101,99,97,99,104,101,95,99,117,95
.b8 50,51,55,57,54,48,98,56,56,97,116,111,109,105,99,79,114,69,80,106,106
.b8 0
.b8 97,116,111,109,105,99,79,114
.b8 0
.b8 7
.b8 185
.b32 497
.b8 12
.b8 6
.b8 144
.b8 177
.b8 200
.b8 201
.b8 171
.b8 2
.b8 2
.b8 97,100,100,114,101,115,115
.b8 0
.b8 7
.b8 185
.b32 1793
.b8 12
.b8 5
.b8 144
.b8 177
.b8 228
.b8 149
.b8 1
.b8 2
.b8 118,97,108
.b8 0
.b8 7
.b8 185
.b32 497
.b8 0
.b8 13
.b64 $L__func_begin1
.b64 $L__func_end1
.b8 1
.b8 156
.b8 95,95,98,111,117,110,100,115,95,99,104,101,99,107,95,115,97,102,101,99,117,100,97
.b8 0
.b8 95,95,98,111,117,110,100,115,95,99,104,101,99,107,95,115,97,102,101,99,117,100,97
.b8 0
.b8 5
.b8 34
.b32 1787
.b8 1
.b8 12
.b8 6
.b8 144
.b8 181
.b8 200
.b8 201
.b8 171
.b8 2
.b8 2
.b8 112,116,114
.b8 0
.b8 5
.b8 34
.b32 1802
.b8 14
.b64 $L__tmp2
.b64 $L__tmp59
.b8 15
.b8 6
.b8 144
.b8 177
.b8 200
.b8 201
.b8 171
.b8 2
.b8 2
.b8 109,101,116,97
.b8 0
.b8 5
.b8 37
.b32 1811
.b8 15
.b8 6
.b8 144
.b8 181
.b8 200
.b8 201
.b8 171
.b8 2
.b8 2
.b8 97,100,100,114
.b8 0
.b8 5
.b8 76
.b32 1825
.b8 16
.b32 .debug_loc
.b8 102,114,101,101,100
.b8 0
.b8 5
.b8 77
.b32 165
.b8 16
.b32 .debug_loc+286
.b8 105,100,120
.b8 0
.b8 5
.b8 78
.b32 758
.b8 15
.b8 6
.b8 144
.b8 177
.b8 228
.b8 200
.b8 171
.b8 2
.b8 2
.b8 111,108,100
.b8 0
.b8 5
.b8 108
.b32 1830
.b8 14
.b64 $L__tmp4
.b64 $L__tmp17
.b8 17
.b8 6
.b8 11
.b8 3
.b64 __local_depot1
.b8 35
.b8 0
.b8 101,110,116,114,121
.b8 0
.b8 5
.b8 41
.b32 313
.b8 15
.b8 6
.b8 144
.b8 181
.b8 200
.b8 201
.b8 171
.b8 2
.b8 2
.b8 97,100,100,114
.b8 0
.b8 5
.b8 45
.b32 1825
.b8 15
.b8 6
.b8 144
.b8 178
.b8 226
.b8 200
.b8 171
.b8 2
.b8 2
.b8 111,108,100
.b8 0
.b8 5
.b8 65
.b32 1830
.b8 14
.b64 $L__tmp8
.b64 $L__tmp12
.b8 15
.b8 6
.b8 144
.b8 182
.b8 226
.b8 200
.b8 171
.b8 2
.b8 2
.b8 111,108,100
.b8 0
.b8 5
.b8 51
.b32 1830
.b8 0
.b8 0
.b8 14
.b64 $L__tmp20
.b64 $L__tmp50
.b8 16
.b32 .debug_loc+561
.b8 105
.b8 0
.b8 5
.b8 80
.b32 463
.b8 14
.b64 $L__tmp27
.b64 $L__tmp44
.b8 15
.b8 6
.b8 144
.b8 180
.b8 200
.b8 201
.b8 171
.b8 2
.b8 2
.b8 101,110,116,114,121
.b8 0
.b8 5
.b8 81
.b32 313
.b8 0
.b8 0
.b8 14
.b64 $L__tmp51
.b64 $L__tmp55
.b8 15
.b8 6
.b8 144
.b8 179
.b8 228
.b8 200
.b8 171
.b8 2
.b8 2
.b8 111,108,100
.b8 0
.b8 5
.b8 98
.b32 1830
.b8 0
.b8 0
.b8 0
.b8 13
.b64 $L__func_begin2
.b64 $L__func_end2
.b8 1
.b8 156
.b8 95,95,98,111,117,110,100,115,95,99,104,101,99,107,95,115,97,102,101,99,117,100,97,95,110,111,95,116,114,97,112
.b8 0
.b8 95,95,98,111,117,110,100,115,95,99,104,101,99,107,95,115,97,102,101,99,117,100,97,95,110,111,95,116,114,97,112
.b8 0
.b8 5
.b8 119
.b32 1787
.b8 1
.b8 12
.b8 6
.b8 144
.b8 181
.b8 200
.b8 201
.b8 171
.b8 2
.b8 2
.b8 112,116,114
.b8 0
.b8 5
.b8 119
.b32 1802
.b8 14
.b64 $L__tmp60
.b64 $L__tmp116
.b8 15
.b8 6
.b8 144
.b8 177
.b8 200
.b8 201
.b8 171
.b8 2
.b8 2
.b8 109,101,116,97
.b8 0
.b8 5
.b8 122
.b32 1811
.b8 15
.b8 6
.b8 144
.b8 181
.b8 200
.b8 201
.b8 171
.b8 2
.b8 2
.b8 97,100,100,114
.b8 0
.b8 5
.b8 161
.b32 1825
.b8 16
.b32 .debug_loc+694
.b8 102,114,101,101,100
.b8 0
.b8 5
.b8 162
.b32 165
.b8 16
.b32 .debug_loc+980
.b8 105,100,120
.b8 0
.b8 5
.b8 163
.b32 758
.b8 15
.b8 6
.b8 144
.b8 177
.b8 228
.b8 200
.b8 171
.b8 2
.b8 2
.b8 111,108,100
.b8 0
.b8 5
.b8 193
.b32 1830
.b8 14
.b64 $L__tmp62
.b64 $L__tmp74
.b8 17
.b8 6
.b8 11
.b8 3
.b64 __local_depot2
.b8 35
.b8 0
.b8 101,110,116,114,121
.b8 0
.b8 5
.b8 126
.b32 313
.b8 15
.b8 6
.b8 144
.b8 181
.b8 200
.b8 201
.b8 171
.b8 2
.b8 2
.b8 97,100,100,114
.b8 0
.b8 5
.b8 131
.b32 1825
.b8 15
.b8 6
.b8 144
.b8 178
.b8 226
.b8 200
.b8 171
.b8 2
.b8 2
.b8 111,108,100
.b8 0
.b8 5
.b8 151
.b32 1830
.b8 14
.b64 $L__tmp66
.b64 $L__tmp69
.b8 15
.b8 6
.b8 144
.b8 182
.b8 226
.b8 200
.b8 171
.b8 2
.b8 2
.b8 111,108,100
.b8 0
.b8 5
.b8 138
.b32 1830
.b8 0
.b8 0
.b8 14
.b64 $L__tmp77
.b64 $L__tmp107
.b8 16
.b32 .debug_loc+1255
.b8 105
.b8 0
.b8 5
.b8 164
.b32 463
.b8 14
.b64 $L__tmp84
.b64 $L__tmp101
.b8 15
.b8 6
.b8 144
.b8 180
.b8 200
.b8 201
.b8 171
.b8 2
.b8 2
.b8 101,110,116,114,121
.b8 0
.b8 5
.b8 165
.b32 313
.b8 0
.b8 0
.b8 14
.b64 $L__tmp108
.b64 $L__tmp112
.b8 15
.b8 6
.b8 144
.b8 179
.b8 228
.b8 200
.b8 171
.b8 2
.b8 2
.b8 111,108,100
.b8 0
.b8 5
.b8 183
.b32 1830
.b8 0
.b8 0
.b8 0
.b8 18
.b8 118,111,105,100
.b8 0
.b8 4
.b32 497
.b32 12
.b8 4
.b32 1787
.b32 12
.b8 4
.b32 1820
.b32 12
.b8 19
.b32 614
.b8 19
.b32 429
.b8 19
.b32 497
.b8 0
	}
	.section	.debug_macinfo
	{
.b8 0

	}

)ptx";

#endif

/**
 * @brief Extract address register from tokenized PTX instruction lexemes
 * @param lexemes Vector of tokenized instruction components
 * @return Register name (e.g., "%rd4") or empty string if not found
 */
static std::string
extractAddressRegister(const std::vector<std::string> &lexemes)
{
	for (size_t i = 0; i < lexemes.size() - 1; ++i) {
		if (lexemes[i] == "[" && i + 1 < lexemes.size()) {
			const std::string &reg = lexemes[i + 1];
			if (!reg.empty() && reg[0] == '%') {
				return reg;
			}
		}
	}
	return "";
}

/**
 * @brief Generate bounds check call instruction with proper indentation
 * @param address_reg Register containing memory address to check
 * @param indentation Whitespace string to match original formatting
 * @param fail_fast Boolean indicating whether to trap kernel or not
 * @return Formatted PTX bounds check instruction
 */
static std::string generateBoundsCheckCall(const std::string &address_reg,
					   const std::string &indentation,
					   const bool fail_fast)
{
	if (fail_fast)
		return indentation + "call __bounds_check_safecuda, (" +
		       address_reg + ");";
	return indentation + "call __bounds_check_safecuda_no_trap, (" +
	       address_reg + ");";
}

/**
 * @brief Extract indentation whitespace from original PTX line
 * @param line Original PTX instruction line
 * @return Leading whitespace string
 */
static std::string getIndentation(const std::string &line)
{
	size_t first_non_space = line.find_first_not_of(" \t");
	if (first_non_space == std::string::npos) {
		return line;
	}
	return line.substr(0, first_non_space);
}

/**
 * @brief Log informational message with cyan color if verbose/debug enabled
 * @param opts SafeCUDA options containing logging flags
 * @param msg Message to display
 */
static void logInfo(const sf_nvcc::SafeCudaOptions &opts,
		    const std::string &msg)
{
	if (opts.enable_verbose || opts.enable_debug) {
		std::cout << ACOL(ACOL_C, ACOL_DF) << msg << ACOL_RESET()
			  << std::endl;
	}
}

/**
 * @brief Log success message with green color if verbose/debug enabled
 * @param opts SafeCUDA options containing logging flags
 * @param msg Message to display
 */
static void logSuccess(const sf_nvcc::SafeCudaOptions &opts,
		       const std::string &msg)
{
	if (opts.enable_verbose || opts.enable_debug) {
		std::cout << ACOL(ACOL_G, ACOL_DF) << msg << ACOL_RESET()
			  << std::endl;
	}
}

/**
 * @brief Log error message with red color
 * @param msg Error message to display
 */
static void logError(const std::string &msg)
{
	std::cout << ACOL(ACOL_R, ACOL_DF) << "Error: " << msg << ACOL_RESET()
		  << std::endl;
}

/**
 * @brief Create backup copy of PTX file with .bak suffix
 * @param ptx_path Path to original PTX file
 * @param opts SafeCUDA options for logging
 * @return True if backup created successfully, false otherwise
 */
static bool createBackupFile(const fs::path &ptx_path,
			     const sf_nvcc::SafeCudaOptions &opts)
{
	fs::path backup_path = ptx_path;
	backup_path += ".bak";

	try {
		fs::copy_file(ptx_path, backup_path,
			      fs::copy_options::overwrite_existing);
		logInfo(opts,
			"Created backup: " + backup_path.filename().string());
		return true;
	} catch (const fs::filesystem_error &) {
		return false;
	}
}

/**
 * @brief Perform PTX file instrumentation by inserting bounds check calls
 * @param ptx_path Path to PTX file to modify
 * @param instructions Vector of identified global memory instructions
 * @param opts SafeCUDA options for logging and debugging
 * @param result Stores the modification result
 * @return True if instrumentation successful, false otherwise
 */
static bool
instrumentPTXFile(const fs::path &ptx_path,
		  const std::vector<sf_nvcc::Instruction> &instructions,
		  const sf_nvcc::SafeCudaOptions &opts,
		  sf_nvcc::PtxModificationResult &result)
{
	std::ifstream input_file(ptx_path);
	if (!input_file.is_open()) {
		return false;
	}

	std::string extern_line;
	if (opts.fail_fast) {
		extern_line = trap_ver;
	} else {
		extern_line = no_trap_ver;
	}

	std::vector<std::string> file_lines;
	std::string line;
	while (std::getline(input_file, line)) {
		file_lines.push_back(line);
	}
	input_file.close();

	std::ofstream output_file(ptx_path);
	if (!output_file.is_open()) {
		return false;
	}

	size_t instruction_index = 0;
	size_t instrumented_count = 0;

	for (size_t line_num = 1; line_num <= file_lines.size(); ++line_num) {
		const std::string &current_line = file_lines[line_num - 1];

		if (current_line.starts_with(".address_size")) {
			output_file << current_line << "\n\n"
				    << extern_line << "\n\n";
			continue;
		}

		bool needs_instrumentation = false;
		std::string bounds_check_call;

		if (!(instruction_index < instructions.size() &&
		      instructions[instruction_index].line_number ==
			      static_cast<int64_t>(line_num))) {
			output_file << current_line << "\n";
			continue;
		}

		const auto &instr = instructions[instruction_index];
		std::string address_reg = extractAddressRegister(instr.lexemes);

		if (!address_reg.empty()) {
			std::string indentation = getIndentation(current_line);
			bounds_check_call = generateBoundsCheckCall(
				address_reg, indentation, opts.fail_fast);
			needs_instrumentation = true;
			instrumented_count++;

			if (opts.enable_debug || opts.enable_verbose) {
				std::string instr_preview;
				if (!instr.lexemes.empty()) {
					instr_preview = instr.lexemes[0];
					if (instr.lexemes.size() > 1) {
						instr_preview +=
							" " + instr.lexemes[1];
						if (instr.lexemes.size() > 2) {
							instr_preview += "...";
						}
					}
				}
				logInfo(opts, "Instrumenting line " +
						      std::to_string(line_num) +
						      ": " + instr_preview);
				logInfo(opts,
					"  → Extracted address register: " +
						address_reg);
			}
		}
		instruction_index++;

		if (needs_instrumentation) {
			output_file << bounds_check_call << "\n";
		}
		output_file << current_line << "\n";
	}

	result.instructions_modified = instrumented_count;
	output_file.close();

	return true;
}

sf_nvcc::PtxModificationResult
sf_nvcc::insert_bounds_check(const fs::path &ptx_path,
			     const SafeCudaOptions &sf_opts)
{
	PtxModificationResult result;
	const auto start = std::chrono::steady_clock::now();

	if (!fs::exists(ptx_path)) {
		logError("PTX file not found: " + ptx_path.string());
		return result;
	}

	if (!fs::is_regular_file(ptx_path)) {
		logError("Path is not a regular file: " + ptx_path.string());
		return result;
	}

	if (sf_opts.enable_debug && !createBackupFile(ptx_path, sf_opts)) {
		logError("Failed to create backup file");
		return result;
	}

	logInfo(sf_opts, "Starting PTX modification for: " +
				 ptx_path.filename().string());

	std::vector<Instruction> instructions;
	try {
		instructions = find_all_ptx(ptx_path.string());
	} catch (const std::exception &e) {
		logError("Failed to parse PTX file: " + std::string(e.what()));
		return result;
	}

	logInfo(sf_opts, "Found " + std::to_string(instructions.size()) +
				 " global memory instructions");

	if (instructions.empty()) {
		logInfo(sf_opts,
			"No global memory instructions found - no instrumentation needed");
		result.success = true;
		result.modified_ptx_path = ptx_path.string();
		result.modification_time_ms =
			std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::steady_clock::now() - start)
				.count();
		return result;
	}

	if (!instrumentPTXFile(ptx_path, instructions, sf_opts, result)) {
		logError("Failed to instrument PTX file");
		return result;
	}

	logSuccess(sf_opts, "Successfully instrumented " +
				    std::to_string(instructions.size()) +
				    " instructions");
	result.success = true;
	result.modification_time_ms =
		std::chrono::duration_cast<std::chrono::microseconds>(
			std::chrono::steady_clock::now() - start)
			.count();
	result.modified_ptx_path = ptx_path.string();
	return result;
}

sf_nvcc::PtxModificationResult
sf_nvcc::modify_ptx(const fs::path &ptx_path, const SafeCudaOptions &sf_opts)
{
	PtxModificationResult result;
	result.success = true;
	result.modified_ptx_path = ptx_path.string();
	if (sf_opts.enable_verbose)
		std::cout << std::endl << ptx_path << '\n';
	if (sf_opts.enable_bounds_check) {
		const PtxModificationResult current_res =
			insert_bounds_check(result.modified_ptx_path, sf_opts);
		result.success &= current_res.success;
		result.instructions_modified +=
			current_res.instructions_modified;
		result.modification_time_ms += current_res.modification_time_ms;
		result.modified_ptx_path = current_res.modified_ptx_path;
	}
	if (sf_opts.enable_verbose) {
		std::cout << "Modification on file: \t\t"
			  << result.modified_ptx_path << "\n\tStatus: "
			  << (result.success ? "Success" : "Failed")
			  << "\n\tInstructions Modified: "
			  << result.instructions_modified
			  << "\n\tModification Time(ms): " << std::fixed
			  << result.modification_time_ms / 1000.0f
			  << std::defaultfloat << "\n"
			  << std::endl;
	}

	return result;
}
