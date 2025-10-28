//
// Created by gin on 28/10/25.
//

#ifndef PTX_DATA_H
#define PTX_DATA_H

#include <string>

#ifdef NDEBUG
const inline std::string trap_ver = R"ptx(
.extern .func  (.param .b32 func_retval0) vprintf
(
	.param .b64 vprintf_param_0,
	.param .b64 vprintf_param_1
)
;
.extern .global .align 1 .u8 FREED_MEM_DEV = 1;
.extern .const .align 8 .u64 d_table;
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

const inline std::string no_trap_ver = R"ptx(
.extern .func  (.param .b32 func_retval0) vprintf
(
	.param .b64 vprintf_param_0,
	.param .b64 vprintf_param_1
)
;
.extern .global .align 1 .u8 FREED_MEM_DEV = 1;
.extern .const .align 8 .u64 d_table;
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

const inline std::string trap_ver = R"ptx(
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
.extern .global .align 1 .u8 FREED_MEM_DEV;
.extern .const .align 8 .u64 d_table;
.global .align 1 .b8 $str[6] = {72, 101, 114, 101, 10};
.global .align 1 .b8 $str$1[11] = {72, 101, 114, 101, 50, 9, 9, 37, 112, 10};
.global .align 1 .b8 $str$2[7] = {72, 101, 114, 101, 51, 10};

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

.func __bounds_check_safecuda(
	.param .b64 __bounds_check_safecuda_param_0
)
{
	.local .align 8 .b8 	__local_depot1[8];
	.reg .b64 	%SP;
	.reg .b64 	%SPL;
	.reg .pred 	%p<20>;
	.reg .b16 	%rs<12>;
	.reg .b32 	%r<29>;
	.reg .b64 	%rd<47>;
	.loc	5 35 0
$L__func_begin1:
	.loc	5 35 0


	mov.u64 	%SPL, __local_depot1;
	cvta.local.u64 	%SP, %SPL;
	ld.param.u64 	%rd3, [__bounds_check_safecuda_param_0];
$L__tmp2:
	.loc	5 37 2
	mov.u64 	%rd4, $str;
	cvta.global.u64 	%rd5, %rd4;
	{ // callseq 1, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd5;
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
$L__tmp3:
	.loc	5 78 2
	mov.u16 	%rs6, 0;
	mov.b16 	%rs1, %rs6;
$L__tmp4:
	.loc	5 79 2
	mov.u32 	%r10, -1;
	mov.b32 	%r1, %r10;
$L__tmp5:
	.loc	5 80 2
	{ // callseq 2, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd5;
	.param .b64 param1;
	st.param.b64 	[param1+0], 0;
	.param .b32 retval0;
	call.uni (retval0),
	vprintf,
	(
	param0,
	param1
	);
	ld.param.b32 	%r11, [retval0+0];
	} // callseq 2
	.loc	5 81 2
	mov.u64 	%rd6, d_table;
	cvta.const.u64 	%rd7, %rd6;
	ld.u64 	%rd8, [%rd7];
	st.u64 	[%SP+0], %rd8;
	mov.u64 	%rd9, $str$1;
	cvta.global.u64 	%rd10, %rd9;
	add.u64 	%rd11, %SP, 0;
	{ // callseq 3, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd10;
	.param .b64 param1;
	st.param.b64 	[param1+0], %rd11;
	.param .b32 retval0;
	call.uni (retval0),
	vprintf,
	(
	param0,
	param1
	);
	ld.param.b32 	%r12, [retval0+0];
	} // callseq 3
$L__tmp6:
	.loc	5 82 2
	mov.u32 	%r14, 1;
	mov.b32 	%r2, %r14;
$L__tmp7:
	mov.u16 	%rs9, %rs1;
$L__tmp8:
	mov.u32 	%r25, %r1;
$L__tmp9:
	mov.u32 	%r26, %r2;
$L__tmp10:
	bra.uni 	$L__BB1_1;

$L__BB1_1:
	mov.u32 	%r4, %r26;
	mov.u32 	%r3, %r25;
	mov.u16 	%rs2, %rs9;
$L__tmp11:
	mov.u64 	%rd12, d_table;
$L__tmp12:
	cvta.const.u64 	%rd13, %rd12;
	ld.u64 	%rd14, [%rd13];
	ld.u32 	%r15, [%rd14+8];
	setp.lt.u32 	%p3, %r4, %r15;
	not.pred 	%p4, %p3;
	@%p4 bra 	$L__BB1_12;
	bra.uni 	$L__BB1_2;

$L__BB1_2:
$L__tmp13:
	.loc	5 83 3
	mov.u64 	%rd35, d_table;
	cvta.const.u64 	%rd36, %rd35;
	ld.u64 	%rd37, [%rd36];
	ld.u64 	%rd38, [%rd37];
	cvt.u64.u32 	%rd39, %r4;
	mul.lo.s64 	%rd40, %rd39, 24;
	add.s64 	%rd2, %rd38, %rd40;
$L__tmp14:
	.loc	5 84 3
	mov.u64 	%rd41, $str$2;
	cvta.global.u64 	%rd42, %rd41;
	{ // callseq 8, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd42;
	.param .b64 param1;
	st.param.b64 	[param1+0], 0;
	.param .b32 retval0;
	call.uni (retval0),
	vprintf,
	(
	param0,
	param1
	);
	ld.param.b32 	%r20, [retval0+0];
	} // callseq 8
	.loc	5 85 3
	ld.u64 	%rd43, [%rd2];
	setp.le.u64 	%p12, %rd43, %rd3;
	mov.pred 	%p11, 0;
	not.pred 	%p13, %p12;
	mov.pred 	%p19, %p11;
	@%p13 bra 	$L__BB1_4;
	bra.uni 	$L__BB1_3;

$L__BB1_3:
	.loc	5 86 7
	ld.u64 	%rd44, [%rd2];
	ld.u32 	%r21, [%rd2+8];
	cvt.u64.u32 	%rd45, %r21;
	add.s64 	%rd46, %rd44, %rd45;
	setp.lt.u64 	%p1, %rd3, %rd46;
	mov.pred 	%p19, %p1;
	bra.uni 	$L__BB1_4;

$L__BB1_4:
	mov.pred 	%p2, %p19;
	not.pred 	%p14, %p2;
	mov.u16 	%rs11, %rs2;
$L__tmp15:
	mov.u32 	%r28, %r3;
$L__tmp16:
	@%p14 bra 	$L__BB1_10;
	bra.uni 	$L__BB1_5;

$L__BB1_5:
$L__tmp17:
	.loc	5 88 4
	ld.u32 	%r22, [%rd2+12];
	setp.eq.s32 	%p15, %r22, 0;
	not.pred 	%p16, %p15;
	@%p16 bra 	$L__BB1_7;
	bra.uni 	$L__BB1_6;

$L__BB1_6:
$L__tmp18:
	.loc	5 89 5
	bra.uni 	$L__BB1_19;
$L__tmp19:

$L__BB1_7:
	.loc	5 91 4
	ld.u32 	%r23, [%rd2+12];
	.loc	5 92 8
	and.b32  	%r24, %r23, 2;
	setp.ne.s32 	%p17, %r24, 0;
	not.pred 	%p18, %p17;
	mov.u16 	%rs10, %rs2;
$L__tmp20:
	mov.u32 	%r27, %r3;
$L__tmp21:
	@%p18 bra 	$L__BB1_9;
	bra.uni 	$L__BB1_8;

$L__BB1_8:
$L__tmp22:
	.loc	5 93 5
	mov.u16 	%rs8, 1;
	mov.b16 	%rs3, %rs8;
$L__tmp23:
	.loc	5 94 5
	mov.b32 	%r5, %r4;
$L__tmp24:
	mov.u16 	%rs10, %rs3;
$L__tmp25:
	mov.u32 	%r27, %r5;
$L__tmp26:
	bra.uni 	$L__BB1_9;

$L__BB1_9:
	mov.u32 	%r6, %r27;
	mov.u16 	%rs4, %rs10;
$L__tmp27:
	mov.u16 	%rs11, %rs4;
$L__tmp28:
	mov.u32 	%r28, %r6;
$L__tmp29:
	bra.uni 	$L__BB1_10;
$L__tmp30:

$L__BB1_10:
	.loc	5 82 48
	mov.u32 	%r7, %r28;
	mov.u16 	%rs5, %rs11;
$L__tmp31:
	bra.uni 	$L__BB1_11;

$L__BB1_11:
	add.s32 	%r8, %r4, 1;
$L__tmp32:
	mov.u16 	%rs9, %rs5;
$L__tmp33:
	mov.u32 	%r25, %r7;
$L__tmp34:
	mov.u32 	%r26, %r8;
$L__tmp35:
	bra.uni 	$L__BB1_1;
$L__tmp36:

$L__BB1_12:
	.loc	5 99 2
	and.b16  	%rs7, %rs2, 255;
	setp.ne.s16 	%p5, %rs7, 0;
	not.pred 	%p6, %p5;
	@%p6 bra 	$L__BB1_16;
	bra.uni 	$L__BB1_13;

$L__BB1_13:
$L__tmp37:
	.loc	5 100 3
	mov.u64 	%rd24, d_table;
	cvta.const.u64 	%rd25, %rd24;
	ld.u64 	%rd26, [%rd25];
	ld.u64 	%rd27, [%rd26];
	add.s64 	%rd28, %rd27, 12;
	.loc	5 100 20
	{ // callseq 6, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd28;
	.param .b32 param1;
	st.param.b32 	[param1+0], 2;
	.param .b32 retval0;
	call.uni (retval0),
	_ZN42_INTERNAL_6da04a83_12_safecache_cu_d_table8atomicOrEPjj,
	(
	param0,
	param1
	);
	ld.param.b32 	%r18, [retval0+0];
$L__tmp38:
	} // callseq 6
	.loc	5 102 3
	and.b32  	%r19, %r18, 2;
	setp.eq.s32 	%p9, %r19, 0;
	not.pred 	%p10, %p9;
	@%p10 bra 	$L__BB1_15;
	bra.uni 	$L__BB1_14;

$L__BB1_14:
$L__tmp39:
	.loc	5 103 4
	mov.u64 	%rd29, d_table;
	cvta.const.u64 	%rd30, %rd29;
	ld.u64 	%rd31, [%rd30];
	ld.u64 	%rd32, [%rd31];
	st.u64 	[%rd32], %rd3;
	.loc	5 104 4
	ld.u64 	%rd33, [%rd30];
	ld.u64 	%rd34, [%rd33];
	st.u32 	[%rd34+8], %r3;
	bra.uni 	$L__BB1_15;
$L__tmp40:

$L__BB1_15:
	.loc	5 106 3
	{ // callseq 7, 0
	.reg .b32 temp_param_reg;
	call.uni
	__trap,
	(
	);
	} // callseq 7
	.loc	5 107 3
	bra.uni 	$L__BB1_19;
$L__tmp41:

$L__BB1_16:
	.loc	5 110 2
	mov.u64 	%rd15, d_table;
	cvta.const.u64 	%rd16, %rd15;
	ld.u64 	%rd17, [%rd16];
	ld.u64 	%rd18, [%rd17];
	add.s64 	%rd19, %rd18, 12;
	.loc	5 110 19
	{ // callseq 4, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd19;
	.param .b32 param1;
	st.param.b32 	[param1+0], 1;
	.param .b32 retval0;
	call.uni (retval0),
	_ZN42_INTERNAL_6da04a83_12_safecache_cu_d_table8atomicOrEPjj,
	(
	param0,
	param1
	);
	ld.param.b32 	%r16, [retval0+0];
$L__tmp42:
	} // callseq 4
	.loc	5 113 2
	and.b32  	%r17, %r16, 1;
	setp.eq.s32 	%p7, %r17, 0;
	not.pred 	%p8, %p7;
	@%p8 bra 	$L__BB1_18;
	bra.uni 	$L__BB1_17;

$L__BB1_17:
$L__tmp43:
	.loc	5 114 3
	mov.u64 	%rd20, d_table;
	cvta.const.u64 	%rd21, %rd20;
	ld.u64 	%rd22, [%rd21];
	ld.u64 	%rd23, [%rd22];
	st.u64 	[%rd23], %rd3;
	bra.uni 	$L__BB1_18;
$L__tmp44:

$L__BB1_18:
	.loc	5 118 2
	{ // callseq 5, 0
	.reg .b32 temp_param_reg;
	call.uni
	__trap,
	(
	);
	} // callseq 5
	.loc	5 119 1
	bra.uni 	$L__BB1_19;

$L__BB1_19:
	ret;
$L__tmp45:
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
.file	1 "/usr/include/stdint.h"
	.file	2 "/home/gin/Desktop/SafeCUDA/include/safecache.cuh"
	.file	3 "/usr/include/bits/types.h"
	.file	4 "/usr/include/bits/stdint-uintn.h"
	.file	5 "/home/gin/Desktop/SafeCUDA/src/safecache.cu"
	.file	6 "/usr/include/bits/stdint-intn.h"
	.file	7 "/usr/local/cuda/bin/../targets/x86_64-linux/include/device_atomic_functions.hpp"
	.section	.debug_loc
	{
.b64 $L__tmp4
.b64 $L__tmp7
.b8 6
.b8 0
.b8 144
.b8 177
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp7
.b64 $L__tmp11
.b8 6
.b8 0
.b8 144
.b8 185
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp11
.b64 $L__tmp14
.b8 6
.b8 0
.b8 144
.b8 178
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp14
.b64 $L__tmp19
.b8 7
.b8 0
.b8 144
.b8 177
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp19
.b64 $L__tmp22
.b8 7
.b8 0
.b8 144
.b8 176
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp22
.b64 $L__tmp24
.b8 6
.b8 0
.b8 144
.b8 179
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp24
.b64 $L__tmp26
.b8 7
.b8 0
.b8 144
.b8 176
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp26
.b64 $L__tmp27
.b8 6
.b8 0
.b8 144
.b8 180
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp27
.b64 $L__tmp30
.b8 7
.b8 0
.b8 144
.b8 177
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp30
.b64 $L__tmp32
.b8 6
.b8 0
.b8 144
.b8 181
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp32
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
.b64 $L__tmp5
.b64 $L__tmp8
.b8 5
.b8 0
.b8 144
.b8 177
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
.b64 $L__tmp15
.b8 5
.b8 0
.b8 144
.b8 179
.b8 228
.b8 149
.b8 1
.b64 $L__tmp15
.b64 $L__tmp20
.b8 6
.b8 0
.b8 144
.b8 180
.b8 228
.b8 200
.b8 171
.b8 2
.b64 $L__tmp20
.b64 $L__tmp23
.b8 6
.b8 0
.b8 144
.b8 179
.b8 228
.b8 200
.b8 171
.b8 2
.b64 $L__tmp23
.b64 $L__tmp25
.b8 5
.b8 0
.b8 144
.b8 181
.b8 228
.b8 149
.b8 1
.b64 $L__tmp25
.b64 $L__tmp26
.b8 6
.b8 0
.b8 144
.b8 179
.b8 228
.b8 200
.b8 171
.b8 2
.b64 $L__tmp26
.b64 $L__tmp28
.b8 5
.b8 0
.b8 144
.b8 182
.b8 228
.b8 149
.b8 1
.b64 $L__tmp28
.b64 $L__tmp30
.b8 6
.b8 0
.b8 144
.b8 180
.b8 228
.b8 200
.b8 171
.b8 2
.b64 $L__tmp30
.b64 $L__tmp33
.b8 5
.b8 0
.b8 144
.b8 183
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
.b64 $L__tmp6
.b64 $L__tmp9
.b8 5
.b8 0
.b8 144
.b8 178
.b8 228
.b8 149
.b8 1
.b64 $L__tmp9
.b64 $L__tmp11
.b8 6
.b8 0
.b8 144
.b8 178
.b8 228
.b8 200
.b8 171
.b8 2
.b64 $L__tmp11
.b64 $L__tmp31
.b8 5
.b8 0
.b8 144
.b8 180
.b8 228
.b8 149
.b8 1
.b64 $L__tmp31
.b64 $L__tmp34
.b8 5
.b8 0
.b8 144
.b8 184
.b8 228
.b8 149
.b8 1
.b64 $L__tmp34
.b64 $L__func_end1
.b8 6
.b8 0
.b8 144
.b8 178
.b8 228
.b8 200
.b8 171
.b8 2
.b64 0
.b64 0
.b64 $L__tmp61
.b64 $L__tmp64
.b8 6
.b8 0
.b8 144
.b8 177
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp64
.b64 $L__tmp68
.b8 7
.b8 0
.b8 144
.b8 176
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp68
.b64 $L__tmp71
.b8 6
.b8 0
.b8 144
.b8 178
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp71
.b64 $L__tmp76
.b8 7
.b8 0
.b8 144
.b8 178
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp76
.b64 $L__tmp79
.b8 7
.b8 0
.b8 144
.b8 177
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp79
.b64 $L__tmp81
.b8 6
.b8 0
.b8 144
.b8 179
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp81
.b64 $L__tmp83
.b8 7
.b8 0
.b8 144
.b8 177
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp83
.b64 $L__tmp84
.b8 6
.b8 0
.b8 144
.b8 180
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp84
.b64 $L__tmp87
.b8 7
.b8 0
.b8 144
.b8 178
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp87
.b64 $L__tmp89
.b8 6
.b8 0
.b8 144
.b8 181
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp89
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
.b64 $L__tmp62
.b64 $L__tmp65
.b8 5
.b8 0
.b8 144
.b8 177
.b8 228
.b8 149
.b8 1
.b64 $L__tmp65
.b64 $L__tmp67
.b8 6
.b8 0
.b8 144
.b8 185
.b8 228
.b8 200
.b8 171
.b8 2
.b64 $L__tmp67
.b64 $L__tmp72
.b8 5
.b8 0
.b8 144
.b8 179
.b8 228
.b8 149
.b8 1
.b64 $L__tmp72
.b64 $L__tmp77
.b8 6
.b8 0
.b8 144
.b8 178
.b8 230
.b8 200
.b8 171
.b8 2
.b64 $L__tmp77
.b64 $L__tmp80
.b8 6
.b8 0
.b8 144
.b8 177
.b8 230
.b8 200
.b8 171
.b8 2
.b64 $L__tmp80
.b64 $L__tmp82
.b8 5
.b8 0
.b8 144
.b8 181
.b8 228
.b8 149
.b8 1
.b64 $L__tmp82
.b64 $L__tmp83
.b8 6
.b8 0
.b8 144
.b8 177
.b8 230
.b8 200
.b8 171
.b8 2
.b64 $L__tmp83
.b64 $L__tmp85
.b8 5
.b8 0
.b8 144
.b8 182
.b8 228
.b8 149
.b8 1
.b64 $L__tmp85
.b64 $L__tmp87
.b8 6
.b8 0
.b8 144
.b8 178
.b8 230
.b8 200
.b8 171
.b8 2
.b64 $L__tmp87
.b64 $L__tmp90
.b8 5
.b8 0
.b8 144
.b8 183
.b8 228
.b8 149
.b8 1
.b64 $L__tmp90
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
.b64 $L__tmp63
.b64 $L__tmp66
.b8 5
.b8 0
.b8 144
.b8 178
.b8 228
.b8 149
.b8 1
.b64 $L__tmp66
.b64 $L__tmp68
.b8 6
.b8 0
.b8 144
.b8 176
.b8 230
.b8 200
.b8 171
.b8 2
.b64 $L__tmp68
.b64 $L__tmp88
.b8 5
.b8 0
.b8 144
.b8 180
.b8 228
.b8 149
.b8 1
.b64 $L__tmp88
.b64 $L__tmp91
.b8 5
.b8 0
.b8 144
.b8 184
.b8 228
.b8 149
.b8 1
.b64 $L__tmp91
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
.b8 16
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
.b8 17
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
.b32 1711
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
.b8 25
.b8 4
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
.b8 2
.b8 70,82,69,69,68,95,77,69,77,95,68,69,86
.b8 0
.b32 500
.b8 1
.b8 5
.b8 26
.b8 5
.b8 9
.b8 3
.b64 FREED_MEM_DEV
.b8 70,82,69,69,68,95,77,69,77,95,68,69,86
.b8 0
.b8 7
.b8 98,111,111,108
.b8 0
.b8 2
.b8 1
.b8 6
.b32 526
.b8 95,95,117,105,110,116,49,54,95,116
.b8 0
.b8 3
.b8 40
.b8 7
.b8 117,110,115,105,103,110,101,100,32,115,104,111,114,116
.b8 0
.b8 7
.b8 2
.b8 6
.b32 508
.b8 117,105,110,116,49,54,95,116
.b8 0
.b8 4
.b8 25
.b8 6
.b32 577
.b8 95,95,117,105,110,116,56,95,116
.b8 0
.b8 3
.b8 38
.b8 7
.b8 117,110,115,105,103,110,101,100,32,99,104,97,114
.b8 0
.b8 8
.b8 1
.b8 6
.b32 560
.b8 117,105,110,116,56,95,116
.b8 0
.b8 4
.b8 24
.b8 4
.b8 95,90,78,56,115,97,102,101,99,117,100,97,54,109,101,109,111,114,121,56,77,101,116,97,100,97,116,97,69
.b8 0
.b8 16
.b8 2
.b8 37
.b8 5
.b8 109,97,103,105,99
.b8 0
.b32 544
.b8 2
.b8 38
.b8 2
.b8 35
.b8 0
.b8 5
.b8 112,97,100,100,105,110,103
.b8 0
.b32 694
.b8 2
.b8 39
.b8 2
.b8 35
.b8 2
.b8 5
.b8 101,110,116,114,121
.b8 0
.b32 253
.b8 2
.b8 40
.b8 2
.b8 35
.b8 8
.b8 0
.b8 8
.b32 594
.b8 9
.b32 706
.b8 6
.b8 0
.b8 10
.b8 95,95,65,82,82,65,89,95,83,73,90,69,95,84,89,80,69,95,95
.b8 0
.b8 8
.b8 7
.b8 6
.b32 746
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
.b32 729
.b8 105,110,116,51,50,95,116
.b8 0
.b8 6
.b8 26
.b8 11
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
.b32 1672
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
.b32 437
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
.b8 35
.b32 1666
.b8 1
.b8 12
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
.b8 35
.b32 1695
.b8 14
.b64 $L__tmp2
.b64 $L__tmp44
.b8 15
.b8 6
.b8 11
.b8 3
.b64 __local_depot1
.b8 35
.b8 0
.b8 109,101,116,97
.b8 0
.b8 5
.b8 38
.b32 1681
.b8 16
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
.b8 77
.b32 1704
.b8 17
.b32 .debug_loc
.b8 102,114,101,101,100
.b8 0
.b8 5
.b8 78
.b32 500
.b8 17
.b32 .debug_loc+284
.b8 105,100,120
.b8 0
.b8 5
.b8 79
.b32 753
.b8 16
.b8 6
.b8 144
.b8 179
.b8 226
.b8 200
.b8 171
.b8 2
.b8 2
.b8 111,108,100
.b8 0
.b8 5
.b8 109
.b32 1709
.b8 14
.b64 $L__tmp5
.b64 $L__tmp35
.b8 17
.b32 .debug_loc+559
.b8 105
.b8 0
.b8 5
.b8 81
.b32 403
.b8 14
.b64 $L__tmp12
.b64 $L__tmp29
.b8 16
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
.b8 82
.b32 253
.b8 0
.b8 0
.b8 14
.b64 $L__tmp36
.b64 $L__tmp40
.b8 16
.b8 6
.b8 144
.b8 181
.b8 226
.b8 200
.b8 171
.b8 2
.b8 2
.b8 111,108,100
.b8 0
.b8 5
.b8 99
.b32 1709
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
.b8 120
.b32 1666
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
.b8 120
.b32 1695
.b8 14
.b64 $L__tmp45
.b64 $L__tmp101
.b8 16
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
.b8 123
.b32 1681
.b8 16
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
.b8 162
.b32 1704
.b8 17
.b32 .debug_loc+692
.b8 102,114,101,101,100
.b8 0
.b8 5
.b8 163
.b32 500
.b8 17
.b32 .debug_loc+978
.b8 105,100,120
.b8 0
.b8 5
.b8 164
.b32 753
.b8 16
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
.b8 194
.b32 1709
.b8 14
.b64 $L__tmp47
.b64 $L__tmp59
.b8 15
.b8 6
.b8 11
.b8 3
.b64 __local_depot2
.b8 35
.b8 0
.b8 101,110,116,114,121
.b8 0
.b8 5
.b8 127
.b32 253
.b8 16
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
.b8 132
.b32 1704
.b8 16
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
.b8 152
.b32 1709
.b8 14
.b64 $L__tmp51
.b64 $L__tmp54
.b8 16
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
.b8 139
.b32 1709
.b8 0
.b8 0
.b8 14
.b64 $L__tmp62
.b64 $L__tmp92
.b8 17
.b32 .debug_loc+1253
.b8 105
.b8 0
.b8 5
.b8 165
.b32 403
.b8 14
.b64 $L__tmp69
.b64 $L__tmp86
.b8 16
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
.b8 166
.b32 253
.b8 0
.b8 0
.b8 14
.b64 $L__tmp93
.b64 $L__tmp97
.b8 16
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
.b8 184
.b32 1709
.b8 0
.b8 0
.b8 0
.b8 18
.b8 118,111,105,100
.b8 0
.b8 3
.b32 437
.b32 12
.b8 3
.b32 1690
.b32 12
.b8 19
.b32 609
.b8 3
.b32 1666
.b32 12
.b8 19
.b32 369
.b8 19
.b32 437
.b8 0
	}
	.section	.debug_macinfo
	{
.b8 0

	}

)ptx";

const inline std::string no_trap_ver = R"ptx(
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
.extern .global .align 1 .u8 FREED_MEM_DEV = 1;
.extern .const .align 8 .u64 d_table;
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
	.loc	5 120 0
$L__func_begin2:
	.loc	5 120 0


	mov.u64 	%SPL, __local_depot2;
	cvta.local.u64 	%SP, %SPL;
	ld.param.u64 	%rd5, [__bounds_check_safecuda_no_trap_param_0];
$L__tmp45:
	.loc	5 122 2
	mov.u64 	%rd6, $str;
	cvta.global.u64 	%rd7, %rd6;
	{ // callseq 6, 0
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
	} // callseq 6
	.loc	5 124 3
	add.s64 	%rd1, %rd5, -16;
$L__tmp46:
	.loc	5 126 2
	ld.u16 	%rs6, [%rd1];
	cvt.u32.u16 	%r10, %rs6;
	setp.eq.s32 	%p5, %r10, 23294;
	not.pred 	%p6, %p5;
	@%p6 bra 	$L__BB2_14;
	bra.uni 	$L__BB2_1;

$L__BB2_1:
$L__tmp47:
	.loc	5 127 3
	ld.u64 	%rd8, [%rd1+8];
	mov.b64 	%rd9, %rd8;
	st.u64 	[%SP+0], %rd9;
	.loc	5 129 3
	ld.u64 	%rd10, [%SP+0];
	setp.eq.s64 	%p7, %rd10, 0;
	not.pred 	%p8, %p7;
	@%p8 bra 	$L__BB2_3;
	bra.uni 	$L__BB2_2;

$L__BB2_2:
$L__tmp48:
	.loc	5 130 4
	bra.uni 	$L__BB2_15;
$L__tmp49:

$L__BB2_3:
	.loc	5 134 3
	ld.u64 	%rd11, [%SP+0];
	ld.u64 	%rd12, [%rd11];
	setp.ge.u64 	%p10, %rd5, %rd12;
	mov.pred 	%p9, 0;
	not.pred 	%p11, %p10;
	mov.pred 	%p35, %p9;
	@%p11 bra 	$L__BB2_5;
	bra.uni 	$L__BB2_4;

$L__BB2_4:
	.loc	5 135 7
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
$L__tmp50:
	.loc	5 137 4
	ld.u64 	%rd27, [%SP+0];
	ld.u32 	%r14, [%rd27+12];
	.loc	5 138 8
	and.b32  	%r15, %r14, 2;
	setp.ne.s32 	%p15, %r15, 0;
	not.pred 	%p16, %p15;
	@%p16 bra 	$L__BB2_10;
	bra.uni 	$L__BB2_7;

$L__BB2_7:
$L__tmp51:
	.loc	5 140 6
	mov.u64 	%rd28, d_table;
	cvta.const.u64 	%rd29, %rd28;
	ld.u64 	%rd30, [%rd29];
	ld.u64 	%rd31, [%rd30];
	add.s64 	%rd32, %rd31, 12;
	.loc	5 139 22
	{ // callseq 8, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd32;
	.param .b32 param1;
	st.param.b32 	[param1+0], 2;
	.param .b32 retval0;
	call.uni (retval0),
	_ZN42_INTERNAL_6da04a83_12_safecache_cu_d_table8atomicOrEPjj,
	(
	param0,
	param1
	);
	ld.param.b32 	%r16, [retval0+0];
$L__tmp52:
	} // callseq 8
	.loc	5 143 10
	and.b32  	%r17, %r16, 2;
	.loc	5 144 9
	setp.eq.s32 	%p17, %r17, 0;
	not.pred 	%p18, %p17;
	@%p18 bra 	$L__BB2_9;
	bra.uni 	$L__BB2_8;

$L__BB2_8:
$L__tmp53:
	.loc	5 145 6
	mov.u64 	%rd33, d_table;
	cvta.const.u64 	%rd34, %rd33;
	ld.u64 	%rd35, [%rd34];
	ld.u64 	%rd36, [%rd35];
	st.u64 	[%rd36], %rd5;
	bra.uni 	$L__BB2_9;

$L__BB2_9:
	bra.uni 	$L__BB2_10;
$L__tmp54:

$L__BB2_10:
	.loc	5 149 4
	bra.uni 	$L__BB2_34;
$L__tmp55:

$L__BB2_11:
	.loc	5 153 4
	mov.u64 	%rd18, d_table;
	cvta.const.u64 	%rd19, %rd18;
	ld.u64 	%rd20, [%rd19];
	ld.u64 	%rd21, [%rd20];
	add.s64 	%rd22, %rd21, 12;
	{ // callseq 7, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd22;
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
$L__tmp56:
	} // callseq 7
	.loc	5 155 3
	and.b32  	%r13, %r12, 1;
	setp.eq.s32 	%p13, %r13, 0;
	not.pred 	%p14, %p13;
	@%p14 bra 	$L__BB2_13;
	bra.uni 	$L__BB2_12;

$L__BB2_12:
$L__tmp57:
	.loc	5 156 4
	mov.u64 	%rd23, d_table;
	cvta.const.u64 	%rd24, %rd23;
	ld.u64 	%rd25, [%rd24];
	ld.u64 	%rd26, [%rd25];
	st.u64 	[%rd26], %rd5;
	bra.uni 	$L__BB2_13;
$L__tmp58:

$L__BB2_13:
	.loc	5 159 3
	bra.uni 	$L__BB2_34;
$L__tmp59:

$L__BB2_14:
	.loc	5 161 1
	bra.uni 	$L__BB2_15;

$L__BB2_15:
$L__tmp60:
	.loc	5 163 2
	mov.u16 	%rs7, 0;
	mov.b16 	%rs1, %rs7;
$L__tmp61:
	.loc	5 164 2
	mov.u32 	%r18, -1;
	mov.b32 	%r1, %r18;
$L__tmp62:
	.loc	5 165 2
	mov.u32 	%r19, 1;
	mov.b32 	%r2, %r19;
$L__tmp63:
	mov.u16 	%rs10, %rs1;
$L__tmp64:
	mov.u32 	%r29, %r1;
$L__tmp65:
	mov.u32 	%r30, %r2;
$L__tmp66:
	bra.uni 	$L__BB2_16;

$L__BB2_16:
	mov.u32 	%r4, %r30;
	mov.u32 	%r3, %r29;
	mov.u16 	%rs2, %rs10;
$L__tmp67:
	mov.u64 	%rd37, d_table;
$L__tmp68:
	cvta.const.u64 	%rd38, %rd37;
	ld.u64 	%rd39, [%rd38];
	ld.u32 	%r20, [%rd39+8];
	setp.lt.u32 	%p19, %r4, %r20;
	not.pred 	%p20, %p19;
	@%p20 bra 	$L__BB2_27;
	bra.uni 	$L__BB2_17;

$L__BB2_17:
$L__tmp69:
	.loc	5 166 3
	mov.u64 	%rd60, d_table;
	cvta.const.u64 	%rd61, %rd60;
	ld.u64 	%rd62, [%rd61];
	ld.u64 	%rd63, [%rd62];
	cvt.u64.u32 	%rd64, %r4;
	mul.lo.s64 	%rd65, %rd64, 24;
	add.s64 	%rd4, %rd63, %rd65;
$L__tmp70:
	.loc	5 168 3
	ld.u64 	%rd66, [%rd4];
	setp.le.u64 	%p28, %rd66, %rd5;
	mov.pred 	%p27, 0;
	not.pred 	%p29, %p28;
	mov.pred 	%p36, %p27;
	@%p29 bra 	$L__BB2_19;
	bra.uni 	$L__BB2_18;

$L__BB2_18:
	.loc	5 169 7
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
$L__tmp71:
	mov.u32 	%r32, %r3;
$L__tmp72:
	@%p30 bra 	$L__BB2_25;
	bra.uni 	$L__BB2_20;

$L__BB2_20:
$L__tmp73:
	.loc	5 171 4
	ld.u32 	%r26, [%rd4+12];
	setp.eq.s32 	%p31, %r26, 0;
	not.pred 	%p32, %p31;
	@%p32 bra 	$L__BB2_22;
	bra.uni 	$L__BB2_21;

$L__BB2_21:
$L__tmp74:
	.loc	5 172 5
	bra.uni 	$L__BB2_34;
$L__tmp75:

$L__BB2_22:
	.loc	5 175 4
	ld.u32 	%r27, [%rd4+12];
	.loc	5 176 8
	and.b32  	%r28, %r27, 2;
	setp.ne.s32 	%p33, %r28, 0;
	not.pred 	%p34, %p33;
	mov.u16 	%rs11, %rs2;
$L__tmp76:
	mov.u32 	%r31, %r3;
$L__tmp77:
	@%p34 bra 	$L__BB2_24;
	bra.uni 	$L__BB2_23;

$L__BB2_23:
$L__tmp78:
	.loc	5 177 5
	mov.u16 	%rs9, 1;
	mov.b16 	%rs3, %rs9;
$L__tmp79:
	.loc	5 178 5
	mov.b32 	%r5, %r4;
$L__tmp80:
	mov.u16 	%rs11, %rs3;
$L__tmp81:
	mov.u32 	%r31, %r5;
$L__tmp82:
	bra.uni 	$L__BB2_24;

$L__BB2_24:
	mov.u32 	%r6, %r31;
	mov.u16 	%rs4, %rs11;
$L__tmp83:
	mov.u16 	%rs12, %rs4;
$L__tmp84:
	mov.u32 	%r32, %r6;
$L__tmp85:
	bra.uni 	$L__BB2_25;
$L__tmp86:

$L__BB2_25:
	.loc	5 165 48
	mov.u32 	%r7, %r32;
	mov.u16 	%rs5, %rs12;
$L__tmp87:
	bra.uni 	$L__BB2_26;

$L__BB2_26:
	add.s32 	%r8, %r4, 1;
$L__tmp88:
	mov.u16 	%rs10, %rs5;
$L__tmp89:
	mov.u32 	%r29, %r7;
$L__tmp90:
	mov.u32 	%r30, %r8;
$L__tmp91:
	bra.uni 	$L__BB2_16;
$L__tmp92:

$L__BB2_27:
	.loc	5 183 2
	and.b16  	%rs8, %rs2, 255;
	setp.ne.s16 	%p21, %rs8, 0;
	not.pred 	%p22, %p21;
	@%p22 bra 	$L__BB2_31;
	bra.uni 	$L__BB2_28;

$L__BB2_28:
$L__tmp93:
	.loc	5 184 3
	mov.u64 	%rd49, d_table;
	cvta.const.u64 	%rd50, %rd49;
	ld.u64 	%rd51, [%rd50];
	ld.u64 	%rd52, [%rd51];
	add.s64 	%rd53, %rd52, 12;
	.loc	5 184 20
	{ // callseq 10, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd53;
	.param .b32 param1;
	st.param.b32 	[param1+0], 2;
	.param .b32 retval0;
	call.uni (retval0),
	_ZN42_INTERNAL_6da04a83_12_safecache_cu_d_table8atomicOrEPjj,
	(
	param0,
	param1
	);
	ld.param.b32 	%r23, [retval0+0];
$L__tmp94:
	} // callseq 10
	.loc	5 186 3
	and.b32  	%r24, %r23, 2;
	setp.eq.s32 	%p25, %r24, 0;
	not.pred 	%p26, %p25;
	@%p26 bra 	$L__BB2_30;
	bra.uni 	$L__BB2_29;

$L__BB2_29:
$L__tmp95:
	.loc	5 187 4
	mov.u64 	%rd54, d_table;
	cvta.const.u64 	%rd55, %rd54;
	ld.u64 	%rd56, [%rd55];
	ld.u64 	%rd57, [%rd56];
	st.u64 	[%rd57], %rd5;
	.loc	5 188 4
	ld.u64 	%rd58, [%rd55];
	ld.u64 	%rd59, [%rd58];
	st.u32 	[%rd59+8], %r3;
	bra.uni 	$L__BB2_30;
$L__tmp96:

$L__BB2_30:
	.loc	5 191 3
	bra.uni 	$L__BB2_34;
$L__tmp97:

$L__BB2_31:
	.loc	5 194 2
	mov.u64 	%rd40, d_table;
	cvta.const.u64 	%rd41, %rd40;
	ld.u64 	%rd42, [%rd41];
	ld.u64 	%rd43, [%rd42];
	add.s64 	%rd44, %rd43, 12;
	.loc	5 194 19
	{ // callseq 9, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd44;
	.param .b32 param1;
	st.param.b32 	[param1+0], 1;
	.param .b32 retval0;
	call.uni (retval0),
	_ZN42_INTERNAL_6da04a83_12_safecache_cu_d_table8atomicOrEPjj,
	(
	param0,
	param1
	);
	ld.param.b32 	%r21, [retval0+0];
$L__tmp98:
	} // callseq 9
	.loc	5 197 2
	and.b32  	%r22, %r21, 1;
	setp.eq.s32 	%p23, %r22, 0;
	not.pred 	%p24, %p23;
	@%p24 bra 	$L__BB2_33;
	bra.uni 	$L__BB2_32;

$L__BB2_32:
$L__tmp99:
	.loc	5 198 3
	mov.u64 	%rd45, d_table;
	cvta.const.u64 	%rd46, %rd45;
	ld.u64 	%rd47, [%rd46];
	ld.u64 	%rd48, [%rd47];
	st.u64 	[%rd48], %rd5;
	bra.uni 	$L__BB2_33;
$L__tmp100:

$L__BB2_33:
	.loc	5 201 1
	bra.uni 	$L__BB2_34;

$L__BB2_34:
	ret;
$L__tmp101:
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
	.file	1 "/usr/include/stdint.h"
	.file	2 "/home/gin/Desktop/SafeCUDA/include/safecache.cuh"
	.file	3 "/usr/include/bits/types.h"
	.file	4 "/usr/include/bits/stdint-uintn.h"
	.file	5 "/home/gin/Desktop/SafeCUDA/src/safecache.cu"
	.file	6 "/usr/include/bits/stdint-intn.h"
	.file	7 "/usr/local/cuda/bin/../targets/x86_64-linux/include/device_atomic_functions.hpp"
	.section	.debug_loc
	{
.b64 $L__tmp4
.b64 $L__tmp7
.b8 6
.b8 0
.b8 144
.b8 177
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp7
.b64 $L__tmp11
.b8 6
.b8 0
.b8 144
.b8 185
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp11
.b64 $L__tmp14
.b8 6
.b8 0
.b8 144
.b8 178
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp14
.b64 $L__tmp19
.b8 7
.b8 0
.b8 144
.b8 177
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp19
.b64 $L__tmp22
.b8 7
.b8 0
.b8 144
.b8 176
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp22
.b64 $L__tmp24
.b8 6
.b8 0
.b8 144
.b8 179
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp24
.b64 $L__tmp26
.b8 7
.b8 0
.b8 144
.b8 176
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp26
.b64 $L__tmp27
.b8 6
.b8 0
.b8 144
.b8 180
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp27
.b64 $L__tmp30
.b8 7
.b8 0
.b8 144
.b8 177
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp30
.b64 $L__tmp32
.b8 6
.b8 0
.b8 144
.b8 181
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp32
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
.b64 $L__tmp5
.b64 $L__tmp8
.b8 5
.b8 0
.b8 144
.b8 177
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
.b64 $L__tmp15
.b8 5
.b8 0
.b8 144
.b8 179
.b8 228
.b8 149
.b8 1
.b64 $L__tmp15
.b64 $L__tmp20
.b8 6
.b8 0
.b8 144
.b8 180
.b8 228
.b8 200
.b8 171
.b8 2
.b64 $L__tmp20
.b64 $L__tmp23
.b8 6
.b8 0
.b8 144
.b8 179
.b8 228
.b8 200
.b8 171
.b8 2
.b64 $L__tmp23
.b64 $L__tmp25
.b8 5
.b8 0
.b8 144
.b8 181
.b8 228
.b8 149
.b8 1
.b64 $L__tmp25
.b64 $L__tmp26
.b8 6
.b8 0
.b8 144
.b8 179
.b8 228
.b8 200
.b8 171
.b8 2
.b64 $L__tmp26
.b64 $L__tmp28
.b8 5
.b8 0
.b8 144
.b8 182
.b8 228
.b8 149
.b8 1
.b64 $L__tmp28
.b64 $L__tmp30
.b8 6
.b8 0
.b8 144
.b8 180
.b8 228
.b8 200
.b8 171
.b8 2
.b64 $L__tmp30
.b64 $L__tmp33
.b8 5
.b8 0
.b8 144
.b8 183
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
.b64 $L__tmp6
.b64 $L__tmp9
.b8 5
.b8 0
.b8 144
.b8 178
.b8 228
.b8 149
.b8 1
.b64 $L__tmp9
.b64 $L__tmp11
.b8 6
.b8 0
.b8 144
.b8 178
.b8 228
.b8 200
.b8 171
.b8 2
.b64 $L__tmp11
.b64 $L__tmp31
.b8 5
.b8 0
.b8 144
.b8 180
.b8 228
.b8 149
.b8 1
.b64 $L__tmp31
.b64 $L__tmp34
.b8 5
.b8 0
.b8 144
.b8 184
.b8 228
.b8 149
.b8 1
.b64 $L__tmp34
.b64 $L__func_end1
.b8 6
.b8 0
.b8 144
.b8 178
.b8 228
.b8 200
.b8 171
.b8 2
.b64 0
.b64 0
.b64 $L__tmp61
.b64 $L__tmp64
.b8 6
.b8 0
.b8 144
.b8 177
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp64
.b64 $L__tmp68
.b8 7
.b8 0
.b8 144
.b8 176
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp68
.b64 $L__tmp71
.b8 6
.b8 0
.b8 144
.b8 178
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp71
.b64 $L__tmp76
.b8 7
.b8 0
.b8 144
.b8 178
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp76
.b64 $L__tmp79
.b8 7
.b8 0
.b8 144
.b8 177
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp79
.b64 $L__tmp81
.b8 6
.b8 0
.b8 144
.b8 179
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp81
.b64 $L__tmp83
.b8 7
.b8 0
.b8 144
.b8 177
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp83
.b64 $L__tmp84
.b8 6
.b8 0
.b8 144
.b8 180
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp84
.b64 $L__tmp87
.b8 7
.b8 0
.b8 144
.b8 178
.b8 226
.b8 204
.b8 147
.b8 215
.b8 4
.b64 $L__tmp87
.b64 $L__tmp89
.b8 6
.b8 0
.b8 144
.b8 181
.b8 230
.b8 201
.b8 171
.b8 2
.b64 $L__tmp89
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
.b64 $L__tmp62
.b64 $L__tmp65
.b8 5
.b8 0
.b8 144
.b8 177
.b8 228
.b8 149
.b8 1
.b64 $L__tmp65
.b64 $L__tmp67
.b8 6
.b8 0
.b8 144
.b8 185
.b8 228
.b8 200
.b8 171
.b8 2
.b64 $L__tmp67
.b64 $L__tmp72
.b8 5
.b8 0
.b8 144
.b8 179
.b8 228
.b8 149
.b8 1
.b64 $L__tmp72
.b64 $L__tmp77
.b8 6
.b8 0
.b8 144
.b8 178
.b8 230
.b8 200
.b8 171
.b8 2
.b64 $L__tmp77
.b64 $L__tmp80
.b8 6
.b8 0
.b8 144
.b8 177
.b8 230
.b8 200
.b8 171
.b8 2
.b64 $L__tmp80
.b64 $L__tmp82
.b8 5
.b8 0
.b8 144
.b8 181
.b8 228
.b8 149
.b8 1
.b64 $L__tmp82
.b64 $L__tmp83
.b8 6
.b8 0
.b8 144
.b8 177
.b8 230
.b8 200
.b8 171
.b8 2
.b64 $L__tmp83
.b64 $L__tmp85
.b8 5
.b8 0
.b8 144
.b8 182
.b8 228
.b8 149
.b8 1
.b64 $L__tmp85
.b64 $L__tmp87
.b8 6
.b8 0
.b8 144
.b8 178
.b8 230
.b8 200
.b8 171
.b8 2
.b64 $L__tmp87
.b64 $L__tmp90
.b8 5
.b8 0
.b8 144
.b8 183
.b8 228
.b8 149
.b8 1
.b64 $L__tmp90
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
.b64 $L__tmp63
.b64 $L__tmp66
.b8 5
.b8 0
.b8 144
.b8 178
.b8 228
.b8 149
.b8 1
.b64 $L__tmp66
.b64 $L__tmp68
.b8 6
.b8 0
.b8 144
.b8 176
.b8 230
.b8 200
.b8 171
.b8 2
.b64 $L__tmp68
.b64 $L__tmp88
.b8 5
.b8 0
.b8 144
.b8 180
.b8 228
.b8 149
.b8 1
.b64 $L__tmp88
.b64 $L__tmp91
.b8 5
.b8 0
.b8 144
.b8 184
.b8 228
.b8 149
.b8 1
.b64 $L__tmp91
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
.b8 16
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
.b8 17
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
.b32 1711
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
.b8 25
.b8 4
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
.b8 2
.b8 70,82,69,69,68,95,77,69,77,95,68,69,86
.b8 0
.b32 500
.b8 1
.b8 5
.b8 26
.b8 5
.b8 9
.b8 3
.b64 FREED_MEM_DEV
.b8 70,82,69,69,68,95,77,69,77,95,68,69,86
.b8 0
.b8 7
.b8 98,111,111,108
.b8 0
.b8 2
.b8 1
.b8 6
.b32 526
.b8 95,95,117,105,110,116,49,54,95,116
.b8 0
.b8 3
.b8 40
.b8 7
.b8 117,110,115,105,103,110,101,100,32,115,104,111,114,116
.b8 0
.b8 7
.b8 2
.b8 6
.b32 508
.b8 117,105,110,116,49,54,95,116
.b8 0
.b8 4
.b8 25
.b8 6
.b32 577
.b8 95,95,117,105,110,116,56,95,116
.b8 0
.b8 3
.b8 38
.b8 7
.b8 117,110,115,105,103,110,101,100,32,99,104,97,114
.b8 0
.b8 8
.b8 1
.b8 6
.b32 560
.b8 117,105,110,116,56,95,116
.b8 0
.b8 4
.b8 24
.b8 4
.b8 95,90,78,56,115,97,102,101,99,117,100,97,54,109,101,109,111,114,121,56,77,101,116,97,100,97,116,97,69
.b8 0
.b8 16
.b8 2
.b8 37
.b8 5
.b8 109,97,103,105,99
.b8 0
.b32 544
.b8 2
.b8 38
.b8 2
.b8 35
.b8 0
.b8 5
.b8 112,97,100,100,105,110,103
.b8 0
.b32 694
.b8 2
.b8 39
.b8 2
.b8 35
.b8 2
.b8 5
.b8 101,110,116,114,121
.b8 0
.b32 253
.b8 2
.b8 40
.b8 2
.b8 35
.b8 8
.b8 0
.b8 8
.b32 594
.b8 9
.b32 706
.b8 6
.b8 0
.b8 10
.b8 95,95,65,82,82,65,89,95,83,73,90,69,95,84,89,80,69,95,95
.b8 0
.b8 8
.b8 7
.b8 6
.b32 746
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
.b32 729
.b8 105,110,116,51,50,95,116
.b8 0
.b8 6
.b8 26
.b8 11
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
.b32 1672
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
.b32 437
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
.b8 35
.b32 1666
.b8 1
.b8 12
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
.b8 35
.b32 1695
.b8 14
.b64 $L__tmp2
.b64 $L__tmp44
.b8 15
.b8 6
.b8 11
.b8 3
.b64 __local_depot1
.b8 35
.b8 0
.b8 109,101,116,97
.b8 0
.b8 5
.b8 38
.b32 1681
.b8 16
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
.b8 77
.b32 1704
.b8 17
.b32 .debug_loc
.b8 102,114,101,101,100
.b8 0
.b8 5
.b8 78
.b32 500
.b8 17
.b32 .debug_loc+284
.b8 105,100,120
.b8 0
.b8 5
.b8 79
.b32 753
.b8 16
.b8 6
.b8 144
.b8 179
.b8 226
.b8 200
.b8 171
.b8 2
.b8 2
.b8 111,108,100
.b8 0
.b8 5
.b8 109
.b32 1709
.b8 14
.b64 $L__tmp5
.b64 $L__tmp35
.b8 17
.b32 .debug_loc+559
.b8 105
.b8 0
.b8 5
.b8 81
.b32 403
.b8 14
.b64 $L__tmp12
.b64 $L__tmp29
.b8 16
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
.b8 82
.b32 253
.b8 0
.b8 0
.b8 14
.b64 $L__tmp36
.b64 $L__tmp40
.b8 16
.b8 6
.b8 144
.b8 181
.b8 226
.b8 200
.b8 171
.b8 2
.b8 2
.b8 111,108,100
.b8 0
.b8 5
.b8 99
.b32 1709
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
.b8 120
.b32 1666
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
.b8 120
.b32 1695
.b8 14
.b64 $L__tmp45
.b64 $L__tmp101
.b8 16
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
.b8 123
.b32 1681
.b8 16
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
.b8 162
.b32 1704
.b8 17
.b32 .debug_loc+692
.b8 102,114,101,101,100
.b8 0
.b8 5
.b8 163
.b32 500
.b8 17
.b32 .debug_loc+978
.b8 105,100,120
.b8 0
.b8 5
.b8 164
.b32 753
.b8 16
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
.b8 194
.b32 1709
.b8 14
.b64 $L__tmp47
.b64 $L__tmp59
.b8 15
.b8 6
.b8 11
.b8 3
.b64 __local_depot2
.b8 35
.b8 0
.b8 101,110,116,114,121
.b8 0
.b8 5
.b8 127
.b32 253
.b8 16
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
.b8 132
.b32 1704
.b8 16
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
.b8 152
.b32 1709
.b8 14
.b64 $L__tmp51
.b64 $L__tmp54
.b8 16
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
.b8 139
.b32 1709
.b8 0
.b8 0
.b8 14
.b64 $L__tmp62
.b64 $L__tmp92
.b8 17
.b32 .debug_loc+1253
.b8 105
.b8 0
.b8 5
.b8 165
.b32 403
.b8 14
.b64 $L__tmp69
.b64 $L__tmp86
.b8 16
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
.b8 166
.b32 253
.b8 0
.b8 0
.b8 14
.b64 $L__tmp93
.b64 $L__tmp97
.b8 16
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
.b8 184
.b32 1709
.b8 0
.b8 0
.b8 0
.b8 18
.b8 118,111,105,100
.b8 0
.b8 3
.b32 437
.b32 12
.b8 3
.b32 1690
.b32 12
.b8 19
.b32 609
.b8 3
.b32 1666
.b32 12
.b8 19
.b32 369
.b8 19
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
