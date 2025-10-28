//
// Created by gin on 28/10/25.
//

#ifndef PTX_DATA_H
#define PTX_DATA_H

#include <string>

#ifdef NDEBUG
const inline std::string trap_ver = R"ptx(
.visible .global .align 8 .u64 d_table;

.visible .func __bounds_check_safecuda(
	.param .b64 __bounds_check_safecuda_param_0
)
{
	.reg .pred 	%p<12>;
	.reg .b16 	%rs<11>;
	.reg .b32 	%r<24>;
	.reg .b64 	%rd<17>;


	ld.param.u64 	%rd6, [__bounds_check_safecuda_param_0];
	ld.global.u64 	%rd7, [d_table];
	add.s64 	%rd1, %rd7, 8;
	ld.u32 	%r1, [%rd7+8];
	setp.lt.u32 	%p1, %r1, 2;
	ld.u64 	%rd2, [%rd7];
	mov.u32 	%r22, -1;
	mov.u16 	%rs9, 0;
	@%p1 bra 	$L__BB0_7;

	mov.u16 	%rs9, 0;
	mov.u32 	%r22, -1;
	mov.u32 	%r20, 1;

$L__BB0_2:
	cvt.u64.u32 	%rd3, %r20;
	mul.wide.u32 	%rd8, %r20, 24;
	add.s64 	%rd4, %rd2, %rd8;
	ld.u64 	%rd5, [%rd4];
	setp.gt.u64 	%p2, %rd5, %rd6;
	@%p2 bra 	$L__BB0_6;

	ld.u32 	%rd9, [%rd4+8];
	add.s64 	%rd10, %rd5, %rd9;
	setp.le.u64 	%p3, %rd10, %rd6;
	@%p3 bra 	$L__BB0_6;

	ld.u32 	%r4, [%rd4+12];
	setp.eq.s32 	%p4, %r4, 0;
	@%p4 bra 	$L__BB0_14;

	cvt.u32.u64 	%r12, %rd3;
	shr.u32 	%r13, %r4, 1;
	and.b32  	%r14, %r13, 1;
	setp.eq.b32 	%p5, %r14, 1;
	selp.b16 	%rs9, 1, %rs9, %p5;
	selp.b32 	%r22, %r12, %r22, %p5;

$L__BB0_6:
	cvt.u32.u64 	%r15, %rd3;
	add.s32 	%r20, %r15, 1;
	setp.lt.u32 	%p6, %r20, %r1;
	@%p6 bra 	$L__BB0_2;

$L__BB0_7:
	and.b16  	%rs7, %rs9, 255;
	setp.eq.s16 	%p7, %rs7, 0;
	@%p7 bra 	$L__BB0_11;

	add.s64 	%rd11, %rd2, 12;
	atom.or.b32 	%r16, [%rd11], 2;
	and.b32  	%r17, %r16, 2;
	setp.ne.s32 	%p8, %r17, 0;
	@%p8 bra 	$L__BB0_10;

	ld.u64 	%rd12, [%rd1+-8];
	st.u64 	[%rd12], %rd6;
	ld.global.u64 	%rd13, [d_table];
	ld.u64 	%rd14, [%rd13];
	st.u32 	[%rd14+8], %r22;

$L__BB0_10:
	// begin inline asm
	trap;
	// end inline asm
	bra.uni 	$L__BB0_14;

$L__BB0_11:
	add.s64 	%rd15, %rd2, 12;
	atom.or.b32 	%r18, [%rd15], 1;
	and.b32  	%r19, %r18, 1;
	setp.eq.b32 	%p9, %r19, 1;
	mov.pred 	%p10, 0;
	xor.pred  	%p11, %p9, %p10;
	@%p11 bra 	$L__BB0_13;

	ld.u64 	%rd16, [%rd1+-8];
	st.u64 	[%rd16], %rd6;

$L__BB0_13:
	// begin inline asm
	trap;
	// end inline asm

$L__BB0_14:
	ret;

}

)ptx";

const inline std::string no_trap_ver = R"ptx(
.visible .global .align 8 .u64 d_table;

.visible .func __bounds_check_safecuda_no_trap(
	.param .b64 __bounds_check_safecuda_no_trap_param_0
)
{
	.reg .pred 	%p<12>;
	.reg .b16 	%rs<11>;
	.reg .b32 	%r<24>;
	.reg .b64 	%rd<17>;


	ld.param.u64 	%rd6, [__bounds_check_safecuda_no_trap_param_0];
	ld.global.u64 	%rd7, [d_table];
	add.s64 	%rd1, %rd7, 8;
	ld.u32 	%r1, [%rd7+8];
	setp.lt.u32 	%p1, %r1, 2;
	ld.u64 	%rd2, [%rd7];
	mov.u32 	%r22, -1;
	mov.u16 	%rs9, 0;
	@%p1 bra 	$L__BB0_7;

	mov.u16 	%rs9, 0;
	mov.u32 	%r22, -1;
	mov.u32 	%r20, 1;

$L__BB0_2:
	cvt.u64.u32 	%rd3, %r20;
	mul.wide.u32 	%rd8, %r20, 24;
	add.s64 	%rd4, %rd2, %rd8;
	ld.u64 	%rd5, [%rd4];
	setp.gt.u64 	%p2, %rd5, %rd6;
	@%p2 bra 	$L__BB0_6;

	ld.u32 	%rd9, [%rd4+8];
	add.s64 	%rd10, %rd5, %rd9;
	setp.le.u64 	%p3, %rd10, %rd6;
	@%p3 bra 	$L__BB0_6;

	ld.u32 	%r4, [%rd4+12];
	setp.eq.s32 	%p4, %r4, 0;
	@%p4 bra 	$L__BB0_12;

	cvt.u32.u64 	%r12, %rd3;
	shr.u32 	%r13, %r4, 1;
	and.b32  	%r14, %r13, 1;
	setp.eq.b32 	%p5, %r14, 1;
	selp.b16 	%rs9, 1, %rs9, %p5;
	selp.b32 	%r22, %r12, %r22, %p5;

$L__BB0_6:
	cvt.u32.u64 	%r15, %rd3;
	add.s32 	%r20, %r15, 1;
	setp.lt.u32 	%p6, %r20, %r1;
	@%p6 bra 	$L__BB0_2;

$L__BB0_7:
	and.b16  	%rs7, %rs9, 255;
	setp.eq.s16 	%p7, %rs7, 0;
	@%p7 bra 	$L__BB0_10;

	add.s64 	%rd11, %rd2, 12;
	atom.or.b32 	%r16, [%rd11], 2;
	and.b32  	%r17, %r16, 2;
	setp.ne.s32 	%p8, %r17, 0;
	@%p8 bra 	$L__BB0_12;

	ld.u64 	%rd12, [%rd1+-8];
	st.u64 	[%rd12], %rd6;
	ld.global.u64 	%rd13, [d_table];
	ld.u64 	%rd14, [%rd13];
	st.u32 	[%rd14+8], %r22;
	bra.uni 	$L__BB0_12;

$L__BB0_10:
	add.s64 	%rd15, %rd2, 12;
	atom.or.b32 	%r18, [%rd15], 1;
	and.b32  	%r19, %r18, 1;
	setp.eq.b32 	%p9, %r19, 1;
	mov.pred 	%p10, 0;
	xor.pred  	%p11, %p9, %p10;
	@%p11 bra 	$L__BB0_12;

	ld.u64 	%rd16, [%rd1+-8];
	st.u64 	[%rd16], %rd6;

$L__BB0_12:
	ret;

}

)ptx";

#else

const inline std::string trap_ver = R"ptx(
.func __trap
()
;
.func  (.param .b32 func_retval0) __uAtomicOr
(
	.param .b64 __uAtomicOr_param_0,
	.param .b32 __uAtomicOr_param_1
)
;
.visible .global .align 8 .u64 d_table;

.func  (.param .b32 func_retval0) _ZN42_INTERNAL_6da04a83_12_safecache_cu_d_table8atomicOrEPjj(
	.param .b64 _ZN42_INTERNAL_6da04a83_12_safecache_cu_d_table8atomicOrEPjj_param_0,
	.param .b32 _ZN42_INTERNAL_6da04a83_12_safecache_cu_d_table8atomicOrEPjj_param_1
)
{
	.reg .b32 	%r<4>;
	.reg .b64 	%rd<3>;
	.loc	7 185 0
$L__func_begin0:
	.loc	7 185 0


	ld.param.u64 	%rd1, [_ZN42_INTERNAL_6da04a83_12_safecache_cu_d_table8atomicOrEPjj_param_0];
	ld.param.u32 	%r1, [_ZN42_INTERNAL_6da04a83_12_safecache_cu_d_table8atomicOrEPjj_param_1];
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
	// .globl	__bounds_check_safecuda
.visible .func __bounds_check_safecuda(
	.param .b64 __bounds_check_safecuda_param_0
)
{
	.reg .pred 	%p<20>;
	.reg .b16 	%rs<12>;
	.reg .b32 	%r<24>;
	.reg .b64 	%rd<37>;
	.loc	5 25 0
$L__func_begin1:
	.loc	5 25 0


	ld.param.u64 	%rd3, [__bounds_check_safecuda_param_0];
$L__tmp2:
	.loc	5 28 2
	mov.u16 	%rs6, 0;
	mov.b16 	%rs1, %rs6;
$L__tmp3:
	.loc	5 29 2
	mov.u32 	%r9, -1;
	mov.b32 	%r1, %r9;
$L__tmp4:
	.loc	5 31 2
	mov.u32 	%r10, 1;
	mov.b32 	%r2, %r10;
$L__tmp5:
	mov.u16 	%rs9, %rs1;
$L__tmp6:
	mov.u32 	%r20, %r1;
$L__tmp7:
	mov.u32 	%r21, %r2;
$L__tmp8:
	bra.uni 	$L__BB1_1;

$L__BB1_1:
	mov.u32 	%r4, %r21;
	mov.u32 	%r3, %r20;
	mov.u16 	%rs2, %rs9;
$L__tmp9:
	mov.u64 	%rd4, d_table;
$L__tmp10:
	cvta.global.u64 	%rd5, %rd4;
	ld.u64 	%rd6, [%rd5];
	ld.u32 	%r11, [%rd6+8];
	setp.lt.u32 	%p3, %r4, %r11;
	not.pred 	%p4, %p3;
	@%p4 bra 	$L__BB1_12;
	bra.uni 	$L__BB1_2;

$L__BB1_2:
$L__tmp11:
	.loc	5 32 3
	mov.u64 	%rd27, d_table;
	cvta.global.u64 	%rd28, %rd27;
	ld.u64 	%rd29, [%rd28];
	ld.u64 	%rd30, [%rd29];
	cvt.u64.u32 	%rd31, %r4;
	mul.lo.s64 	%rd32, %rd31, 24;
	add.s64 	%rd2, %rd30, %rd32;
$L__tmp12:
	.loc	5 34 3
	ld.u64 	%rd33, [%rd2];
	setp.le.u64 	%p12, %rd33, %rd3;
	mov.pred 	%p11, 0;
	not.pred 	%p13, %p12;
	mov.pred 	%p19, %p11;
	@%p13 bra 	$L__BB1_4;
	bra.uni 	$L__BB1_3;

$L__BB1_3:
	.loc	5 35 7
	ld.u64 	%rd34, [%rd2];
	ld.u32 	%r16, [%rd2+8];
	cvt.u64.u32 	%rd35, %r16;
	add.s64 	%rd36, %rd34, %rd35;
	setp.lt.u64 	%p1, %rd3, %rd36;
	mov.pred 	%p19, %p1;
	bra.uni 	$L__BB1_4;

$L__BB1_4:
	mov.pred 	%p2, %p19;
	not.pred 	%p14, %p2;
	mov.u16 	%rs11, %rs2;
$L__tmp13:
	mov.u32 	%r23, %r3;
$L__tmp14:
	@%p14 bra 	$L__BB1_10;
	bra.uni 	$L__BB1_5;

$L__BB1_5:
$L__tmp15:
	.loc	5 37 4
	ld.u32 	%r17, [%rd2+12];
	setp.eq.s32 	%p15, %r17, 0;
	not.pred 	%p16, %p15;
	@%p16 bra 	$L__BB1_7;
	bra.uni 	$L__BB1_6;

$L__BB1_6:
$L__tmp16:
	.loc	5 38 5
	bra.uni 	$L__BB1_19;
$L__tmp17:

$L__BB1_7:
	.loc	5 40 4
	ld.u32 	%r18, [%rd2+12];
	.loc	5 41 8
	and.b32  	%r19, %r18, 2;
	setp.ne.s32 	%p17, %r19, 0;
	not.pred 	%p18, %p17;
	mov.u16 	%rs10, %rs2;
$L__tmp18:
	mov.u32 	%r22, %r3;
$L__tmp19:
	@%p18 bra 	$L__BB1_9;
	bra.uni 	$L__BB1_8;

$L__BB1_8:
$L__tmp20:
	.loc	5 42 5
	mov.u16 	%rs8, 1;
	mov.b16 	%rs3, %rs8;
$L__tmp21:
	.loc	5 43 5
	mov.b32 	%r5, %r4;
$L__tmp22:
	mov.u16 	%rs10, %rs3;
$L__tmp23:
	mov.u32 	%r22, %r5;
$L__tmp24:
	bra.uni 	$L__BB1_9;

$L__BB1_9:
	mov.u32 	%r6, %r22;
	mov.u16 	%rs4, %rs10;
$L__tmp25:
	mov.u16 	%rs11, %rs4;
$L__tmp26:
	mov.u32 	%r23, %r6;
$L__tmp27:
	bra.uni 	$L__BB1_10;
$L__tmp28:

$L__BB1_10:
	.loc	5 31 48
	mov.u32 	%r7, %r23;
	mov.u16 	%rs5, %rs11;
$L__tmp29:
	bra.uni 	$L__BB1_11;

$L__BB1_11:
	add.s32 	%r8, %r4, 1;
$L__tmp30:
	mov.u16 	%rs9, %rs5;
$L__tmp31:
	mov.u32 	%r20, %r7;
$L__tmp32:
	mov.u32 	%r21, %r8;
$L__tmp33:
	bra.uni 	$L__BB1_1;
$L__tmp34:

$L__BB1_12:
	.loc	5 48 2
	and.b16  	%rs7, %rs2, 255;
	setp.ne.s16 	%p5, %rs7, 0;
	not.pred 	%p6, %p5;
	@%p6 bra 	$L__BB1_16;
	bra.uni 	$L__BB1_13;

$L__BB1_13:
$L__tmp35:
	.loc	5 49 3
	mov.u64 	%rd16, d_table;
	cvta.global.u64 	%rd17, %rd16;
	ld.u64 	%rd18, [%rd17];
	ld.u64 	%rd19, [%rd18];
	add.s64 	%rd20, %rd19, 12;
	.loc	5 49 20
	{ // callseq 3, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd20;
	.param .b32 param1;
	st.param.b32 	[param1+0], 2;
	.param .b32 retval0;
	call.uni (retval0),
	_ZN42_INTERNAL_6da04a83_12_safecache_cu_d_table8atomicOrEPjj,
	(
	param0,
	param1
	);
	ld.param.b32 	%r14, [retval0+0];
$L__tmp36:
	} // callseq 3
	.loc	5 51 3
	and.b32  	%r15, %r14, 2;
	setp.eq.s32 	%p9, %r15, 0;
	not.pred 	%p10, %p9;
	@%p10 bra 	$L__BB1_15;
	bra.uni 	$L__BB1_14;

$L__BB1_14:
$L__tmp37:
	.loc	5 52 4
	mov.u64 	%rd21, d_table;
	cvta.global.u64 	%rd22, %rd21;
	ld.u64 	%rd23, [%rd22];
	ld.u64 	%rd24, [%rd23];
	st.u64 	[%rd24], %rd3;
	.loc	5 53 4
	ld.u64 	%rd25, [%rd22];
	ld.u64 	%rd26, [%rd25];
	st.u32 	[%rd26+8], %r3;
	bra.uni 	$L__BB1_15;
$L__tmp38:

$L__BB1_15:
	.loc	5 55 3
	{ // callseq 4, 0
	.reg .b32 temp_param_reg;
	call.uni
	__trap,
	(
	);
	} // callseq 4
	.loc	5 56 3
	bra.uni 	$L__BB1_19;
$L__tmp39:

$L__BB1_16:
	.loc	5 59 2
	mov.u64 	%rd7, d_table;
	cvta.global.u64 	%rd8, %rd7;
	ld.u64 	%rd9, [%rd8];
	ld.u64 	%rd10, [%rd9];
	add.s64 	%rd11, %rd10, 12;
	.loc	5 59 19
	{ // callseq 1, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd11;
	.param .b32 param1;
	st.param.b32 	[param1+0], 1;
	.param .b32 retval0;
	call.uni (retval0),
	_ZN42_INTERNAL_6da04a83_12_safecache_cu_d_table8atomicOrEPjj,
	(
	param0,
	param1
	);
	ld.param.b32 	%r12, [retval0+0];
$L__tmp40:
	} // callseq 1
	.loc	5 62 2
	and.b32  	%r13, %r12, 1;
	setp.eq.s32 	%p7, %r13, 0;
	not.pred 	%p8, %p7;
	@%p8 bra 	$L__BB1_18;
	bra.uni 	$L__BB1_17;

$L__BB1_17:
$L__tmp41:
	.loc	5 63 3
	mov.u64 	%rd12, d_table;
	cvta.global.u64 	%rd13, %rd12;
	ld.u64 	%rd14, [%rd13];
	ld.u64 	%rd15, [%rd14];
	st.u64 	[%rd15], %rd3;
	bra.uni 	$L__BB1_18;
$L__tmp42:

$L__BB1_18:
	.loc	5 67 2
	{ // callseq 2, 0
	.reg .b32 temp_param_reg;
	call.uni
	__trap,
	(
	);
	} // callseq 2
	.loc	5 68 1
	bra.uni 	$L__BB1_19;

$L__BB1_19:
	ret;
$L__tmp43:
$L__func_end1:

}
.func __trap()
{



	// begin inline asm
	trap;
	// end inline asm
	ret;
$L__func_end2:

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
$L__func_end3:

}
	.file	1 "/usr/include/stdint.h"
	.file	2 "/home/gin/Desktop/SafeCUDA/include/safecache.cuh"
	.file	3 "/usr/include/bits/types.h"
	.file	4 "/usr/include/bits/stdint-uintn.h"
	.file	5 "/home/gin/Desktop/SafeCUDA/src/safecache.cu"
	.file	6 "/usr/include/bits/stdint-intn.h"
	.file	7 "/usr/local/cuda/bin/../targets/x86_64-linux/include/device_atomic_functions.hpp"
	.section	.debug_loc
	{
.b64 $L__tmp3
.b64 $L__tmp6
.b8 6
.b8 0
.b8 144
.b8 177
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp6
.b64 $L__tmp10
.b8 6
.b8 0
.b8 144
.b8 185
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp10
.b64 $L__tmp13
.b8 6
.b8 0
.b8 144
.b8 178
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp13
.b64 $L__tmp18
.b8 7
.b8 0
.b8 144
.b8 177
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp18
.b64 $L__tmp21
.b8 7
.b8 0
.b8 144
.b8 176
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp21
.b64 $L__tmp23
.b8 6
.b8 0
.b8 144
.b8 179
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp23
.b64 $L__tmp25
.b8 7
.b8 0
.b8 144
.b8 176
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp25
.b64 $L__tmp26
.b8 6
.b8 0
.b8 144
.b8 180
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp26
.b64 $L__tmp29
.b8 7
.b8 0
.b8 144
.b8 177
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp29
.b64 $L__tmp31
.b8 6
.b8 0
.b8 144
.b8 181
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp31
.b64 $L__func_end1
.b8 6
.b8 0
.b8 144
.b8 185
.b8 230
.b8 201
.b8 171
.b8 2
.b64 0
.b64 0
.b64 $L__tmp4
.b64 $L__tmp7
.b8 5
.b8 0
.b8 144
.b8 177
.b8 228
.b8 149
.b8 1
.b64 $L__tmp7
.b64 $L__tmp9
.b8 6
.b8 0
.b8 144
.b8 176
.b8 228
.b8 200
.b8 171
.b8 2
.b64 $L__tmp9
.b64 $L__tmp14
.b8 5
.b8 0
.b8 144
.b8 179
.b8 228
.b8 149
.b8 1
.b64 $L__tmp14
.b64 $L__tmp19
.b8 6
.b8 0
.b8 144
.b8 179
.b8 228
.b8 200
.b8 171
.b8 2
.b64 $L__tmp19
.b64 $L__tmp22
.b8 6
.b8 0
.b8 144
.b8 178
.b8 228
.b8 200
.b8 171
.b8 2
.b64 $L__tmp22
.b64 $L__tmp24
.b8 5
.b8 0
.b8 144
.b8 181
.b8 228
.b8 149
.b8 1
.b64 $L__tmp24
.b64 $L__tmp25
.b8 6
.b8 0
.b8 144
.b8 178
.b8 228
.b8 200
.b8 171
.b8 2
.b64 $L__tmp25
.b64 $L__tmp27
.b8 5
.b8 0
.b8 144
.b8 182
.b8 228
.b8 149
.b8 1
.b64 $L__tmp27
.b64 $L__tmp29
.b8 6
.b8 0
.b8 144
.b8 179
.b8 228
.b8 200
.b8 171
.b8 2
.b64 $L__tmp29
.b64 $L__tmp32
.b8 5
.b8 0
.b8 144
.b8 183
.b8 228
.b8 149
.b8 1
.b64 $L__tmp32
.b64 $L__func_end1
.b8 6
.b8 0
.b8 144
.b8 176
.b8 228
.b8 200
.b8 171
.b8 2
.b64 0
.b64 0
.b64 $L__tmp5
.b64 $L__tmp8
.b8 5
.b8 0
.b8 144
.b8 178
.b8 228
.b8 149
.b8 1
.b64 $L__tmp8
.b64 $L__tmp10
.b8 6
.b8 0
.b8 144
.b8 177
.b8 228
.b8 200
.b8 171
.b8 2
.b64 $L__tmp10
.b64 $L__tmp30
.b8 5
.b8 0
.b8 144
.b8 180
.b8 228
.b8 149
.b8 1
.b64 $L__tmp30
.b64 $L__tmp33
.b8 5
.b8 0
.b8 144
.b8 184
.b8 228
.b8 149
.b8 1
.b64 $L__tmp33
.b64 $L__func_end1
.b8 6
.b8 0
.b8 144
.b8 177
.b8 228
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
.b8 15
.b8 0
.b8 73
.b8 19
.b8 51
.b8 6
.b8 0
.b8 0
.b8 4
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
.b8 5
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
.b8 6
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
.b8 7
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
.b8 8
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
.b8 9
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
.b8 10
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
.b8 11
.b8 11
.b8 1
.b8 17
.b8 1
.b8 18
.b8 1
.b8 0
.b8 0
.b8 12
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
.b8 13
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
.b8 14
.b8 59
.b8 0
.b8 3
.b8 8
.b8 0
.b8 0
.b8 15
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
.b32 958
.b8 2
.b8 0
.b32 .debug_abbrev
.b8 8
.b8 1
.b8 108,103,101,110,102,101,58,32,69,68,71,32,54,46,54
.b8 0
.b8 4
.b8 0
.b8 47,104,111,109,101,47,103,105,110,47,68,101,115,107,116,111,112,47,83,97,102,101,67,85,68,65,47,115,114,99,47,115,97,102,101,99,97,99,104,101
.b8 46,99,117
.b8 0
.b32 .debug_line
.b8 47,104,111,109,101,47,103,105,110,47,68,101,115,107,116,111,112,47,83,97,102,101,67,85,68,65
.b8 0
.b64 0
.b8 2
.b8 100,95,116,97,98,108,101
.b8 0
.b32 148
.b8 1
.b8 5
.b8 23
.b8 5
.b8 9
.b8 3
.b64 d_table
.b8 100,95,116,97,98,108,101
.b8 0
.b8 3
.b32 157
.b32 12
.b8 4
.b8 95,90,78,56,115,97,102,101,99,117,100,97,54,109,101,109,111,114,121,49,53,65,108,108,111,99,97,116,105,111,110,84,97,98,108,101,69
.b8 0
.b8 16
.b8 2
.b8 43
.b8 5
.b8 101,110,116,114,105,101,115
.b8 0
.b32 253
.b8 2
.b8 44
.b8 2
.b8 35
.b8 0
.b8 5
.b8 99,111,117,110,116
.b8 0
.b32 403
.b8 2
.b8 45
.b8 2
.b8 35
.b8 8
.b8 5
.b8 99,97,112,97,99,105,116,121
.b8 0
.b32 403
.b8 2
.b8 46
.b8 2
.b8 35
.b8 12
.b8 0
.b8 3
.b32 262
.b32 12
.b8 4
.b8 95,90,78,56,115,97,102,101,99,117,100,97,54,109,101,109,111,114,121,53,69,110,116,114,121,69
.b8 0
.b8 24
.b8 2
.b8 30
.b8 5
.b8 115,116,97,114,116,95,97,100,100,114
.b8 0
.b32 369
.b8 2
.b8 31
.b8 2
.b8 35
.b8 0
.b8 5
.b8 98,108,111,99,107,95,115,105,122,101
.b8 0
.b32 403
.b8 2
.b8 32
.b8 2
.b8 35
.b8 8
.b8 5
.b8 102,108,97,103,115
.b8 0
.b32 403
.b8 2
.b8 33
.b8 2
.b8 35
.b8 12
.b8 5
.b8 101,112,111,99,104,115
.b8 0
.b32 403
.b8 2
.b8 34
.b8 2
.b8 35
.b8 16
.b8 0
.b8 6
.b32 386
.b8 117,105,110,116,112,116,114,95,116
.b8 0
.b8 1
.b8 79
.b8 7
.b8 117,110,115,105,103,110,101,100,32,108,111,110,103
.b8 0
.b8 7
.b8 8
.b8 6
.b32 419
.b8 117,105,110,116,51,50,95,116
.b8 0
.b8 4
.b8 26
.b8 6
.b32 437
.b8 95,95,117,105,110,116,51,50,95,116
.b8 0
.b8 3
.b8 42
.b8 7
.b8 117,110,115,105,103,110,101,100,32,105,110,116
.b8 0
.b8 7
.b8 4
.b8 6
.b32 470
.b8 95,95,105,110,116,51,50,95,116
.b8 0
.b8 3
.b8 41
.b8 7
.b8 105,110,116
.b8 0
.b8 5
.b8 4
.b8 6
.b32 453
.b8 105,110,116,51,50,95,116
.b8 0
.b8 6
.b8 26
.b8 8
.b64 $L__func_begin0
.b64 $L__func_end0
.b8 1
.b8 156
.b8 95,90,78,52,50,95,73,78,84,69,82,78,65,76,95,54,100,97,48,52,97,56,51,95,49,50,95,115,97,102,101,99,97,99,104,101,95,99,117,95
.b8 100,95,116,97,98,108,101,56,97,116,111,109,105,99,79,114,69,80,106,106
.b8 0
.b8 97,116,111,109,105,99,79,114
.b8 0
.b8 7
.b8 185
.b32 437
.b8 9
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
.b32 925
.b8 9
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
.b32 437
.b8 0
.b8 10
.b64 $L__func_begin1
.b64 $L__func_end1
.b8 1
.b8 156
.b8 95,95,98,111,117,110,100,115,95,99,104,101,99,107,95,115,97,102,101,99,117,100,97
.b8 0
.b8 95,95,98,111,117,110,100,115,95,99,104,101,99,107,95,115,97,102,101,99,117,100,97
.b8 0
.b8 5
.b8 25
.b32 919
.b8 1
.b8 9
.b8 6
.b8 144
.b8 179
.b8 200
.b8 201
.b8 171
.b8 2
.b8 2
.b8 112,116,114
.b8 0
.b8 5
.b8 25
.b32 934
.b8 11
.b64 $L__tmp2
.b64 $L__tmp43
.b8 12
.b8 6
.b8 144
.b8 179
.b8 200
.b8 201
.b8 171
.b8 2
.b8 2
.b8 97,100,100,114
.b8 0
.b8 5
.b8 27
.b32 943
.b8 13
.b32 .debug_loc
.b8 102,114,101,101,100
.b8 0
.b8 5
.b8 28
.b32 948
.b8 13
.b32 .debug_loc+284
.b8 105,100,120
.b8 0
.b8 5
.b8 29
.b32 477
.b8 12
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
.b8 59
.b32 956
.b8 11
.b64 $L__tmp4
.b64 $L__tmp34
.b8 13
.b32 .debug_loc+559
.b8 105
.b8 0
.b8 5
.b8 31
.b32 403
.b8 11
.b64 $L__tmp11
.b64 $L__tmp28
.b8 12
.b8 6
.b8 144
.b8 178
.b8 200
.b8 201
.b8 171
.b8 2
.b8 2
.b8 101,110,116,114,121
.b8 0
.b8 5
.b8 32
.b32 253
.b8 0
.b8 0
.b8 11
.b64 $L__tmp35
.b64 $L__tmp39
.b8 12
.b8 6
.b8 144
.b8 180
.b8 226
.b8 200
.b8 171
.b8 2
.b8 2
.b8 111,108,100
.b8 0
.b8 5
.b8 49
.b32 956
.b8 0
.b8 0
.b8 0
.b8 14
.b8 118,111,105,100
.b8 0
.b8 3
.b32 437
.b32 12
.b8 3
.b32 919
.b32 12
.b8 15
.b32 369
.b8 7
.b8 98,111,111,108
.b8 0
.b8 2
.b8 1
.b8 15
.b32 437
.b8 0
	}
	.section	.debug_macinfo
	{
.b8 0

	}

)ptx";

const inline std::string no_trap_ver = R"ptx(
.func  (.param .b32 func_retval0) __uAtomicOr
(
	.param .b64 __uAtomicOr_param_0,
	.param .b32 __uAtomicOr_param_1
)
;
.visible .global .align 8 .u64 d_table;

.func  (.param .b32 func_retval0) _ZN42_INTERNAL_6da04a83_12_safecache_cu_d_table8atomicOrEPjj(
	.param .b64 _ZN42_INTERNAL_6da04a83_12_safecache_cu_d_table8atomicOrEPjj_param_0,
	.param .b32 _ZN42_INTERNAL_6da04a83_12_safecache_cu_d_table8atomicOrEPjj_param_1
)
{
	.reg .b32 	%r<4>;
	.reg .b64 	%rd<3>;
	.loc	7 185 0
$L__func_begin0:
	.loc	7 185 0


	ld.param.u64 	%rd1, [_ZN42_INTERNAL_6da04a83_12_safecache_cu_d_table8atomicOrEPjj_param_0];
	ld.param.u32 	%r1, [_ZN42_INTERNAL_6da04a83_12_safecache_cu_d_table8atomicOrEPjj_param_1];
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
	// .globl	__bounds_check_safecuda_no_trap
.visible .func __bounds_check_safecuda_no_trap(
	.param .b64 __bounds_check_safecuda_no_trap_param_0
)
{
	.reg .pred 	%p<20>;
	.reg .b16 	%rs<12>;
	.reg .b32 	%r<24>;
	.reg .b64 	%rd<37>;
	.loc	5 25 0
$L__func_begin1:
	.loc	5 25 0


	ld.param.u64 	%rd3, [__bounds_check_safecuda_no_trap_param_0];
$L__tmp2:
	.loc	5 28 2
	mov.u16 	%rs6, 0;
	mov.b16 	%rs1, %rs6;
$L__tmp3:
	.loc	5 29 2
	mov.u32 	%r9, -1;
	mov.b32 	%r1, %r9;
$L__tmp4:
	.loc	5 31 2
	mov.u32 	%r10, 1;
	mov.b32 	%r2, %r10;
$L__tmp5:
	mov.u16 	%rs9, %rs1;
$L__tmp6:
	mov.u32 	%r20, %r1;
$L__tmp7:
	mov.u32 	%r21, %r2;
$L__tmp8:
	bra.uni 	$L__BB1_1;

$L__BB1_1:
	mov.u32 	%r4, %r21;
	mov.u32 	%r3, %r20;
	mov.u16 	%rs2, %rs9;
$L__tmp9:
	mov.u64 	%rd4, d_table;
$L__tmp10:
	cvta.global.u64 	%rd5, %rd4;
	ld.u64 	%rd6, [%rd5];
	ld.u32 	%r11, [%rd6+8];
	setp.lt.u32 	%p3, %r4, %r11;
	not.pred 	%p4, %p3;
	@%p4 bra 	$L__BB1_12;
	bra.uni 	$L__BB1_2;

$L__BB1_2:
$L__tmp11:
	.loc	5 32 3
	mov.u64 	%rd27, d_table;
	cvta.global.u64 	%rd28, %rd27;
	ld.u64 	%rd29, [%rd28];
	ld.u64 	%rd30, [%rd29];
	cvt.u64.u32 	%rd31, %r4;
	mul.lo.s64 	%rd32, %rd31, 24;
	add.s64 	%rd2, %rd30, %rd32;
$L__tmp12:
	.loc	5 34 3
	ld.u64 	%rd33, [%rd2];
	setp.le.u64 	%p12, %rd33, %rd3;
	mov.pred 	%p11, 0;
	not.pred 	%p13, %p12;
	mov.pred 	%p19, %p11;
	@%p13 bra 	$L__BB1_4;
	bra.uni 	$L__BB1_3;

$L__BB1_3:
	.loc	5 35 7
	ld.u64 	%rd34, [%rd2];
	ld.u32 	%r16, [%rd2+8];
	cvt.u64.u32 	%rd35, %r16;
	add.s64 	%rd36, %rd34, %rd35;
	setp.lt.u64 	%p1, %rd3, %rd36;
	mov.pred 	%p19, %p1;
	bra.uni 	$L__BB1_4;

$L__BB1_4:
	mov.pred 	%p2, %p19;
	not.pred 	%p14, %p2;
	mov.u16 	%rs11, %rs2;
$L__tmp13:
	mov.u32 	%r23, %r3;
$L__tmp14:
	@%p14 bra 	$L__BB1_10;
	bra.uni 	$L__BB1_5;

$L__BB1_5:
$L__tmp15:
	.loc	5 37 4
	ld.u32 	%r17, [%rd2+12];
	setp.eq.s32 	%p15, %r17, 0;
	not.pred 	%p16, %p15;
	@%p16 bra 	$L__BB1_7;
	bra.uni 	$L__BB1_6;

$L__BB1_6:
$L__tmp16:
	.loc	5 38 5
	bra.uni 	$L__BB1_19;
$L__tmp17:

$L__BB1_7:
	.loc	5 40 4
	ld.u32 	%r18, [%rd2+12];
	.loc	5 41 8
	and.b32  	%r19, %r18, 2;
	setp.ne.s32 	%p17, %r19, 0;
	not.pred 	%p18, %p17;
	mov.u16 	%rs10, %rs2;
$L__tmp18:
	mov.u32 	%r22, %r3;
$L__tmp19:
	@%p18 bra 	$L__BB1_9;
	bra.uni 	$L__BB1_8;

$L__BB1_8:
$L__tmp20:
	.loc	5 42 5
	mov.u16 	%rs8, 1;
	mov.b16 	%rs3, %rs8;
$L__tmp21:
	.loc	5 43 5
	mov.b32 	%r5, %r4;
$L__tmp22:
	mov.u16 	%rs10, %rs3;
$L__tmp23:
	mov.u32 	%r22, %r5;
$L__tmp24:
	bra.uni 	$L__BB1_9;

$L__BB1_9:
	mov.u32 	%r6, %r22;
	mov.u16 	%rs4, %rs10;
$L__tmp25:
	mov.u16 	%rs11, %rs4;
$L__tmp26:
	mov.u32 	%r23, %r6;
$L__tmp27:
	bra.uni 	$L__BB1_10;
$L__tmp28:

$L__BB1_10:
	.loc	5 31 48
	mov.u32 	%r7, %r23;
	mov.u16 	%rs5, %rs11;
$L__tmp29:
	bra.uni 	$L__BB1_11;

$L__BB1_11:
	add.s32 	%r8, %r4, 1;
$L__tmp30:
	mov.u16 	%rs9, %rs5;
$L__tmp31:
	mov.u32 	%r20, %r7;
$L__tmp32:
	mov.u32 	%r21, %r8;
$L__tmp33:
	bra.uni 	$L__BB1_1;
$L__tmp34:

$L__BB1_12:
	.loc	5 48 2
	and.b16  	%rs7, %rs2, 255;
	setp.ne.s16 	%p5, %rs7, 0;
	not.pred 	%p6, %p5;
	@%p6 bra 	$L__BB1_16;
	bra.uni 	$L__BB1_13;

$L__BB1_13:
$L__tmp35:
	.loc	5 49 3
	mov.u64 	%rd16, d_table;
	cvta.global.u64 	%rd17, %rd16;
	ld.u64 	%rd18, [%rd17];
	ld.u64 	%rd19, [%rd18];
	add.s64 	%rd20, %rd19, 12;
	.loc	5 49 20
	{ // callseq 2, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd20;
	.param .b32 param1;
	st.param.b32 	[param1+0], 2;
	.param .b32 retval0;
	call.uni (retval0),
	_ZN42_INTERNAL_6da04a83_12_safecache_cu_d_table8atomicOrEPjj,
	(
	param0,
	param1
	);
	ld.param.b32 	%r14, [retval0+0];
$L__tmp36:
	} // callseq 2
	.loc	5 51 3
	and.b32  	%r15, %r14, 2;
	setp.eq.s32 	%p9, %r15, 0;
	not.pred 	%p10, %p9;
	@%p10 bra 	$L__BB1_15;
	bra.uni 	$L__BB1_14;

$L__BB1_14:
$L__tmp37:
	.loc	5 52 4
	mov.u64 	%rd21, d_table;
	cvta.global.u64 	%rd22, %rd21;
	ld.u64 	%rd23, [%rd22];
	ld.u64 	%rd24, [%rd23];
	st.u64 	[%rd24], %rd3;
	.loc	5 53 4
	ld.u64 	%rd25, [%rd22];
	ld.u64 	%rd26, [%rd25];
	st.u32 	[%rd26+8], %r3;
	bra.uni 	$L__BB1_15;
$L__tmp38:

$L__BB1_15:
	.loc	5 55 3
	bra.uni 	$L__BB1_19;
$L__tmp39:

$L__BB1_16:
	.loc	5 58 2
	mov.u64 	%rd7, d_table;
	cvta.global.u64 	%rd8, %rd7;
	ld.u64 	%rd9, [%rd8];
	ld.u64 	%rd10, [%rd9];
	add.s64 	%rd11, %rd10, 12;
	.loc	5 58 19
	{ // callseq 1, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd11;
	.param .b32 param1;
	st.param.b32 	[param1+0], 1;
	.param .b32 retval0;
	call.uni (retval0),
	_ZN42_INTERNAL_6da04a83_12_safecache_cu_d_table8atomicOrEPjj,
	(
	param0,
	param1
	);
	ld.param.b32 	%r12, [retval0+0];
$L__tmp40:
	} // callseq 1
	.loc	5 61 2
	and.b32  	%r13, %r12, 1;
	setp.eq.s32 	%p7, %r13, 0;
	not.pred 	%p8, %p7;
	@%p8 bra 	$L__BB1_18;
	bra.uni 	$L__BB1_17;

$L__BB1_17:
$L__tmp41:
	.loc	5 62 3
	mov.u64 	%rd12, d_table;
	cvta.global.u64 	%rd13, %rd12;
	ld.u64 	%rd14, [%rd13];
	ld.u64 	%rd15, [%rd14];
	st.u64 	[%rd15], %rd3;
	bra.uni 	$L__BB1_18;
$L__tmp42:

$L__BB1_18:
	.loc	5 65 1
	bra.uni 	$L__BB1_19;

$L__BB1_19:
	ret;
$L__tmp43:
$L__func_end1:

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
$L__func_end2:

}
	.file	1 "/usr/include/stdint.h"
	.file	2 "/home/gin/Desktop/SafeCUDA/include/safecache.cuh"
	.file	3 "/usr/include/bits/types.h"
	.file	4 "/usr/include/bits/stdint-uintn.h"
	.file	5 "/home/gin/Desktop/SafeCUDA/src/safecache.cu"
	.file	6 "/usr/include/bits/stdint-intn.h"
	.file	7 "/usr/local/cuda/bin/../targets/x86_64-linux/include/device_atomic_functions.hpp"
	.section	.debug_loc
	{
.b64 $L__tmp3
.b64 $L__tmp6
.b8 6
.b8 0
.b8 144
.b8 177
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp6
.b64 $L__tmp10
.b8 6
.b8 0
.b8 144
.b8 185
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp10
.b64 $L__tmp13
.b8 6
.b8 0
.b8 144
.b8 178
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp13
.b64 $L__tmp18
.b8 7
.b8 0
.b8 144
.b8 177
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp18
.b64 $L__tmp21
.b8 7
.b8 0
.b8 144
.b8 176
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp21
.b64 $L__tmp23
.b8 6
.b8 0
.b8 144
.b8 179
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp23
.b64 $L__tmp25
.b8 7
.b8 0
.b8 144
.b8 176
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp25
.b64 $L__tmp26
.b8 6
.b8 0
.b8 144
.b8 180
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp26
.b64 $L__tmp29
.b8 7
.b8 0
.b8 144
.b8 177
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp29
.b64 $L__tmp31
.b8 6
.b8 0
.b8 144
.b8 181
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp31
.b64 $L__func_end1
.b8 6
.b8 0
.b8 144
.b8 185
.b8 230
.b8 201
.b8 171
.b8 2
.b64 0
.b64 0
.b64 $L__tmp4
.b64 $L__tmp7
.b8 5
.b8 0
.b8 144
.b8 177
.b8 228
.b8 149
.b8 1
.b64 $L__tmp7
.b64 $L__tmp9
.b8 6
.b8 0
.b8 144
.b8 176
.b8 228
.b8 200
.b8 171
.b8 2
.b64 $L__tmp9
.b64 $L__tmp14
.b8 5
.b8 0
.b8 144
.b8 179
.b8 228
.b8 149
.b8 1
.b64 $L__tmp14
.b64 $L__tmp19
.b8 6
.b8 0
.b8 144
.b8 179
.b8 228
.b8 200
.b8 171
.b8 2
.b64 $L__tmp19
.b64 $L__tmp22
.b8 6
.b8 0
.b8 144
.b8 178
.b8 228
.b8 200
.b8 171
.b8 2
.b64 $L__tmp22
.b64 $L__tmp24
.b8 5
.b8 0
.b8 144
.b8 181
.b8 228
.b8 149
.b8 1
.b64 $L__tmp24
.b64 $L__tmp25
.b8 6
.b8 0
.b8 144
.b8 178
.b8 228
.b8 200
.b8 171
.b8 2
.b64 $L__tmp25
.b64 $L__tmp27
.b8 5
.b8 0
.b8 144
.b8 182
.b8 228
.b8 149
.b8 1
.b64 $L__tmp27
.b64 $L__tmp29
.b8 6
.b8 0
.b8 144
.b8 179
.b8 228
.b8 200
.b8 171
.b8 2
.b64 $L__tmp29
.b64 $L__tmp32
.b8 5
.b8 0
.b8 144
.b8 183
.b8 228
.b8 149
.b8 1
.b64 $L__tmp32
.b64 $L__func_end1
.b8 6
.b8 0
.b8 144
.b8 176
.b8 228
.b8 200
.b8 171
.b8 2
.b64 0
.b64 0
.b64 $L__tmp5
.b64 $L__tmp8
.b8 5
.b8 0
.b8 144
.b8 178
.b8 228
.b8 149
.b8 1
.b64 $L__tmp8
.b64 $L__tmp10
.b8 6
.b8 0
.b8 144
.b8 177
.b8 228
.b8 200
.b8 171
.b8 2
.b64 $L__tmp10
.b64 $L__tmp30
.b8 5
.b8 0
.b8 144
.b8 180
.b8 228
.b8 149
.b8 1
.b64 $L__tmp30
.b64 $L__tmp33
.b8 5
.b8 0
.b8 144
.b8 184
.b8 228
.b8 149
.b8 1
.b64 $L__tmp33
.b64 $L__func_end1
.b8 6
.b8 0
.b8 144
.b8 177
.b8 228
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
.b8 15
.b8 0
.b8 73
.b8 19
.b8 51
.b8 6
.b8 0
.b8 0
.b8 4
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
.b8 5
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
.b8 6
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
.b8 7
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
.b8 8
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
.b8 9
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
.b8 10
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
.b8 11
.b8 11
.b8 1
.b8 17
.b8 1
.b8 18
.b8 1
.b8 0
.b8 0
.b8 12
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
.b8 13
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
.b8 14
.b8 59
.b8 0
.b8 3
.b8 8
.b8 0
.b8 0
.b8 15
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
.b32 974
.b8 2
.b8 0
.b32 .debug_abbrev
.b8 8
.b8 1
.b8 108,103,101,110,102,101,58,32,69,68,71,32,54,46,54
.b8 0
.b8 4
.b8 0
.b8 47,104,111,109,101,47,103,105,110,47,68,101,115,107,116,111,112,47,83,97,102,101,67,85,68,65,47,115,114,99,47,115,97,102,101,99,97,99,104,101
.b8 46,99,117
.b8 0
.b32 .debug_line
.b8 47,104,111,109,101,47,103,105,110,47,68,101,115,107,116,111,112,47,83,97,102,101,67,85,68,65
.b8 0
.b64 0
.b8 2
.b8 100,95,116,97,98,108,101
.b8 0
.b32 148
.b8 1
.b8 5
.b8 23
.b8 5
.b8 9
.b8 3
.b64 d_table
.b8 100,95,116,97,98,108,101
.b8 0
.b8 3
.b32 157
.b32 12
.b8 4
.b8 95,90,78,56,115,97,102,101,99,117,100,97,54,109,101,109,111,114,121,49,53,65,108,108,111,99,97,116,105,111,110,84,97,98,108,101,69
.b8 0
.b8 16
.b8 2
.b8 43
.b8 5
.b8 101,110,116,114,105,101,115
.b8 0
.b32 253
.b8 2
.b8 44
.b8 2
.b8 35
.b8 0
.b8 5
.b8 99,111,117,110,116
.b8 0
.b32 403
.b8 2
.b8 45
.b8 2
.b8 35
.b8 8
.b8 5
.b8 99,97,112,97,99,105,116,121
.b8 0
.b32 403
.b8 2
.b8 46
.b8 2
.b8 35
.b8 12
.b8 0
.b8 3
.b32 262
.b32 12
.b8 4
.b8 95,90,78,56,115,97,102,101,99,117,100,97,54,109,101,109,111,114,121,53,69,110,116,114,121,69
.b8 0
.b8 24
.b8 2
.b8 30
.b8 5
.b8 115,116,97,114,116,95,97,100,100,114
.b8 0
.b32 369
.b8 2
.b8 31
.b8 2
.b8 35
.b8 0
.b8 5
.b8 98,108,111,99,107,95,115,105,122,101
.b8 0
.b32 403
.b8 2
.b8 32
.b8 2
.b8 35
.b8 8
.b8 5
.b8 102,108,97,103,115
.b8 0
.b32 403
.b8 2
.b8 33
.b8 2
.b8 35
.b8 12
.b8 5
.b8 101,112,111,99,104,115
.b8 0
.b32 403
.b8 2
.b8 34
.b8 2
.b8 35
.b8 16
.b8 0
.b8 6
.b32 386
.b8 117,105,110,116,112,116,114,95,116
.b8 0
.b8 1
.b8 79
.b8 7
.b8 117,110,115,105,103,110,101,100,32,108,111,110,103
.b8 0
.b8 7
.b8 8
.b8 6
.b32 419
.b8 117,105,110,116,51,50,95,116
.b8 0
.b8 4
.b8 26
.b8 6
.b32 437
.b8 95,95,117,105,110,116,51,50,95,116
.b8 0
.b8 3
.b8 42
.b8 7
.b8 117,110,115,105,103,110,101,100,32,105,110,116
.b8 0
.b8 7
.b8 4
.b8 6
.b32 470
.b8 95,95,105,110,116,51,50,95,116
.b8 0
.b8 3
.b8 41
.b8 7
.b8 105,110,116
.b8 0
.b8 5
.b8 4
.b8 6
.b32 453
.b8 105,110,116,51,50,95,116
.b8 0
.b8 6
.b8 26
.b8 8
.b64 $L__func_begin0
.b64 $L__func_end0
.b8 1
.b8 156
.b8 95,90,78,52,50,95,73,78,84,69,82,78,65,76,95,54,100,97,48,52,97,56,51,95,49,50,95,115,97,102,101,99,97,99,104,101,95,99,117,95
.b8 100,95,116,97,98,108,101,56,97,116,111,109,105,99,79,114,69,80,106,106
.b8 0
.b8 97,116,111,109,105,99,79,114
.b8 0
.b8 7
.b8 185
.b32 437
.b8 9
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
.b32 941
.b8 9
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
.b32 437
.b8 0
.b8 10
.b64 $L__func_begin1
.b64 $L__func_end1
.b8 1
.b8 156
.b8 95,95,98,111,117,110,100,115,95,99,104,101,99,107,95,115,97,102,101,99,117,100,97,95,110,111,95,116,114,97,112
.b8 0
.b8 95,95,98,111,117,110,100,115,95,99,104,101,99,107,95,115,97,102,101,99,117,100,97,95,110,111,95,116,114,97,112
.b8 0
.b8 5
.b8 25
.b32 935
.b8 1
.b8 9
.b8 6
.b8 144
.b8 179
.b8 200
.b8 201
.b8 171
.b8 2
.b8 2
.b8 112,116,114
.b8 0
.b8 5
.b8 25
.b32 950
.b8 11
.b64 $L__tmp2
.b64 $L__tmp43
.b8 12
.b8 6
.b8 144
.b8 179
.b8 200
.b8 201
.b8 171
.b8 2
.b8 2
.b8 97,100,100,114
.b8 0
.b8 5
.b8 27
.b32 959
.b8 13
.b32 .debug_loc
.b8 102,114,101,101,100
.b8 0
.b8 5
.b8 28
.b32 964
.b8 13
.b32 .debug_loc+284
.b8 105,100,120
.b8 0
.b8 5
.b8 29
.b32 477
.b8 12
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
.b8 58
.b32 972
.b8 11
.b64 $L__tmp4
.b64 $L__tmp34
.b8 13
.b32 .debug_loc+559
.b8 105
.b8 0
.b8 5
.b8 31
.b32 403
.b8 11
.b64 $L__tmp11
.b64 $L__tmp28
.b8 12
.b8 6
.b8 144
.b8 178
.b8 200
.b8 201
.b8 171
.b8 2
.b8 2
.b8 101,110,116,114,121
.b8 0
.b8 5
.b8 32
.b32 253
.b8 0
.b8 0
.b8 11
.b64 $L__tmp35
.b64 $L__tmp39
.b8 12
.b8 6
.b8 144
.b8 180
.b8 226
.b8 200
.b8 171
.b8 2
.b8 2
.b8 111,108,100
.b8 0
.b8 5
.b8 49
.b32 972
.b8 0
.b8 0
.b8 0
.b8 14
.b8 118,111,105,100
.b8 0
.b8 3
.b32 437
.b32 12
.b8 3
.b32 935
.b32 12
.b8 15
.b32 369
.b8 7
.b8 98,111,111,108
.b8 0
.b8 2
.b8 1
.b8 15
.b32 437
.b8 0
	}
	.section	.debug_macinfo
	{
.b8 0

	}

)ptx";

#endif

#endif //PTX_DATA_H
