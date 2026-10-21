; Hand-authored typed-pointer companion of native_sample_positions.ll.
target datalayout = "e-p:64:64-i64:64-n8:16:32:64"
target triple = "air64_v28-apple-macosx26.0.0"

define <4 x float> @native_position_vertex(i32 %id) {
  %right = icmp eq i32 %id, 1
  %top = icmp eq i32 %id, 2
  %x = select i1 %right, float 3.0, float -1.0
  %y = select i1 %top, float 3.0, float -1.0
  %a = insertelement <4 x float> <float 0.0, float 0.0, float 0.5, float 1.0>, float %x, i32 0
  %b = insertelement <4 x float> %a, float %y, i32 1
  ret <4 x float> %b
}

define <4 x float> @native_sample_point(<2 x float> %point, i32 %sample) {
  %x = extractelement <2 x float> %point, i32 0
  %y = extractelement <2 x float> %point, i32 1
  %s = uitofp i32 %sample to float
  %a = insertelement <4 x float> <float 0.0, float 0.0, float 0.0, float 1.0>, float %x, i32 0
  %b = insertelement <4 x float> %a, float %y, i32 1
  %c = insertelement <4 x float> %b, float %s, i32 2
  ret <4 x float> %c
}

define <4 x float> @native_sample_barycentric(<3 x float> %barycentric, i32 %sample) {
  %x = extractelement <3 x float> %barycentric, i32 0
  %y = extractelement <3 x float> %barycentric, i32 1
  %s = uitofp i32 %sample to float
  %a = insertelement <4 x float> <float 0.0, float 0.0, float 0.0, float 1.0>, float %x, i32 0
  %b = insertelement <4 x float> %a, float %y, i32 1
  %c = insertelement <4 x float> %b, float %s, i32 2
  ret <4 x float> %c
}

!air.version = !{!0}
!air.language_version = !{!1}
!air.vertex = !{!2}
!air.fragment = !{!7, !11, !16}
!0 = !{i32 2, i32 8, i32 0}
!1 = !{!"Metal", i32 4, i32 0, i32 0}
!2 = !{<4 x float> (i32)* @native_position_vertex, !3, !5}
!3 = !{!4}
!4 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!5 = !{!6}
!6 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"id"}
!7 = !{<4 x float> (<2 x float>, i32)* @native_sample_point, !8, !10}
!8 = !{!9}
!9 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!10 = !{!12, !13}
!11 = !{<4 x float> (<3 x float>, i32)* @native_sample_barycentric, !8, !14}
!12 = !{i32 0, !"air.point_coord", !"air.arg_type_name", !"float2", !"air.arg_name", !"point"}
!13 = !{i32 1, !"air.sample_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"sample"}
!14 = !{!15, !13}
!15 = !{i32 0, !"air.barycentric_coord", !"air.center", !"air.perspective", !"air.arg_type_name", !"float3", !"air.arg_name", !"barycentric"}

; The device test records constant depth/w and invocation counts for each sample.
define void @native_sample_depth(<4 x float> %position, i32 %sample, i32 addrspace(1)* %words) {
  %z = extractelement <4 x float> %position, i32 2
  %w = extractelement <4 x float> %position, i32 3
  %zb = bitcast float %z to i32
  %wb = bitcast float %w to i32
  %zptr = getelementptr i32, i32 addrspace(1)* %words, i32 %sample
  %old_z = call i32 @air.atomic.global.add.u.i32(i32 addrspace(1)* %zptr, i32 %zb, i32 0, i32 2, i1 true)
  %wi = add i32 %sample, 4
  %wptr = getelementptr i32, i32 addrspace(1)* %words, i32 %wi
  %old_w = call i32 @air.atomic.global.add.u.i32(i32 addrspace(1)* %wptr, i32 %wb, i32 0, i32 2, i1 true)
  %ci = add i32 %sample, 8
  %cptr = getelementptr i32, i32 addrspace(1)* %words, i32 %ci
  %old = call i32 @air.atomic.global.add.u.i32(i32 addrspace(1)* %cptr, i32 1, i32 0, i32 2, i1 true)
  ret void
}
declare i32 @air.atomic.global.add.u.i32(i32 addrspace(1)*, i32, i32, i32, i1)
!16 = !{void (<4 x float>, i32, i32 addrspace(1)*)* @native_sample_depth, !17, !18}
!17 = !{}
!18 = !{!19, !13, !20}
!19 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!20 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"words"}
