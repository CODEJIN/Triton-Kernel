import torch
import triton
import triton.ops
import triton.language as tl
import logging

# TODO:
# Basic Conv
    # Forward
    # Backward
    # with Bias
# Stride
    # Forward
    # Backward
# Dilation
    # Forward
    # Backward


def get_cuda_autotune_config_conv_forward():
    return [
        triton.Config({'BLOCK_SIZE_BATCH': 128, 'BLOCK_SIZE_IN_CHANNELS': 256, 'BLOCK_SIZE_OUT_CHANNELS': 64, 'GROUP_SIZE': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_BATCH': 64, 'BLOCK_SIZE_IN_CHANNELS': 256, 'BLOCK_SIZE_OUT_CHANNELS': 32, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_BATCH': 128, 'BLOCK_SIZE_IN_CHANNELS': 128, 'BLOCK_SIZE_OUT_CHANNELS': 32, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_BATCH': 128, 'BLOCK_SIZE_IN_CHANNELS': 64, 'BLOCK_SIZE_OUT_CHANNELS': 32, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_BATCH': 64, 'BLOCK_SIZE_IN_CHANNELS': 128, 'BLOCK_SIZE_OUT_CHANNELS': 32, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_BATCH': 128, 'BLOCK_SIZE_IN_CHANNELS': 32, 'BLOCK_SIZE_OUT_CHANNELS': 32, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_BATCH': 64, 'BLOCK_SIZE_IN_CHANNELS': 32, 'BLOCK_SIZE_OUT_CHANNELS': 32, 'GROUP_SIZE': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_BATCH': 32, 'BLOCK_SIZE_IN_CHANNELS': 64, 'BLOCK_SIZE_OUT_CHANNELS': 32, 'GROUP_SIZE': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_BATCH': 128, 'BLOCK_SIZE_IN_CHANNELS': 256, 'BLOCK_SIZE_OUT_CHANNELS': 128, 'GROUP_SIZE': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_BATCH': 256, 'BLOCK_SIZE_IN_CHANNELS': 128, 'BLOCK_SIZE_OUT_CHANNELS': 128, 'GROUP_SIZE': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_BATCH': 256, 'BLOCK_SIZE_IN_CHANNELS': 64, 'BLOCK_SIZE_OUT_CHANNELS': 128, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_BATCH': 64, 'BLOCK_SIZE_IN_CHANNELS': 256, 'BLOCK_SIZE_OUT_CHANNELS': 128, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_BATCH': 128, 'BLOCK_SIZE_IN_CHANNELS': 128, 'BLOCK_SIZE_OUT_CHANNELS': 128, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_BATCH': 128, 'BLOCK_SIZE_IN_CHANNELS': 64, 'BLOCK_SIZE_OUT_CHANNELS': 64, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_BATCH': 64, 'BLOCK_SIZE_IN_CHANNELS': 128, 'BLOCK_SIZE_OUT_CHANNELS': 64, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_BATCH': 128, 'BLOCK_SIZE_IN_CHANNELS': 32, 'BLOCK_SIZE_OUT_CHANNELS': 64, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4)
    ]

def get_cuda_autotune_config_matmul_forward():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=4,
                      num_warps=4)
    ]


class _Conv1d_without_Bias_Triton_Kernel:
    @staticmethod
    @triton.autotune(
        configs=get_cuda_autotune_config_conv_forward(),
        key=['BATCH', 'IN_CHANNELS', 'IN_LENGTH', 'OUT_CHANNELS', 'KERNEL_SIZE'],
        )
    @triton.jit
    def forward(
        x_pointer,  # [Batch, In_channels, Length]
        weights_pointer,  # [Out_channels, In_channels, Kernel_size]
        y_pointer,  # [Batch, Out_channels, Out_length]
        
        BATCH: int,
        IN_CHANNELS: int,
        IN_LENGTH: int,
        OUT_CHANNELS: int,
        KERNEL_SIZE: int,
        OUT_LENGTH: int,
        
        # Memory strides
        stride_x_batch: int,
        stride_x_in_channels: int,
        stride_x_in_length: int,

        stride_weights_out_channels: int,
        stride_weights_in_channels: int,
        stride_weights_kernel_size: int,

        stride_y_batch: int,
        stride_y_out_channels: int,
        stride_y_out_length: int,
        
        # Block sizes
        BLOCK_SIZE_BATCH: tl.constexpr,
        BLOCK_SIZE_IN_CHANNELS: tl.constexpr,
        BLOCK_SIZE_OUT_CHANNELS: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        
        DTYPE: tl.constexpr
        ):
        '''
        Conv1d 연산에서 BLOCK 단위로 움직일 요소는 다음과 같음
        1. BATCH
        2. IN_CHANNELS
        3. IN_LENGTH
        4. KERNEL_SIZE (always moving 1)

        BLAS를 사용하는 방식은 [M, K] @ [K, N] -> [M, N]임
        CONV 1D에서 이 연산은 [BATCH, IN_CHANNELS] @ [IN_CHANNELS, OUT_CHANNELS] -> [BATCH, OUT_CHANNELS] 임
        그리고 또 하나 고려할 점은 [BATCH, OUT_CHANNELS]라는 출력에서 각 셀은 커널 내에서 완결되어야 함
        CONV의 특성상 모든 커널에 대해서 진행된 BLAS연산의 합이 더해질 필요가 있음
        
        예를 들어 KERNEL_SIZE가 3, STRIDE가 1일때, 첫 [0:BLOCK_SIZE_BATCH, 0:BLOCK_SIZE_OUT_CHANNEL, 0:1]를 계산하기 위해서는 다음의 세 연산이 필요함
            * [0:BLOCK_SIZE_BATCH, 0:BLOCK_SIZE_IN_CHANNELS, 0] @ [0:BLOCK_SIZE_IN_CHANNELS, 0:BLOCK_SIZE_OUT_CHANNELS, 0]
            * [0:BLOCK_SIZE_BATCH, 0:BLOCK_SIZE_IN_CHANNELS, 1] @ [0:BLOCK_SIZE_IN_CHANNELS, 0:BLOCK_SIZE_OUT_CHANNELS, 1]
            * [0:BLOCK_SIZE_BATCH, 0:BLOCK_SIZE_IN_CHANNELS, 2] @ [0:BLOCK_SIZE_IN_CHANNELS, 0:BLOCK_SIZE_OUT_CHANNELS, 2]
        이 세 연산 결과 모두가 더해지는 결과가 [0:BLOCK_SIZE_BATCH, 0:BLOCK_SIZE_OUT_CHANNEL, 0:1]가 됨
        
        이에 따라 커널 내 LOOP 연산은 tl.advance를 통한 다음과 같은 이동이 필요함
            * INPUT x의 경우 단일 커널 내에서 IN_CHANNELS, IN_LENGTH가 움직여야 함
            * WEIGHTS weights의 경우 단일 커널 내에서 IN_CHANNELS, KERNEL_SIZE가 움직여야 함

        tl.make_block_ptr의 초기 offsets은 다음과 같이 설정할 필요가 있음
            * INPUT x의 경우 (pid_batch * BLOCK_SIZE_BATCH, 0, pid_out_length * CONV_STRIDE)
            * WEIGHTS weights의 경우 (pid_out_channels * BLOCK_SIZE_OUT_CHANNELS, 0, 0)

        Loop 내에서 tl.advance의 움직임은 다음과 같이 설정할 필요가 있음
            * IN_CHANNELS 루프마다 (0, BLOCK_SIZE_IN_CHANNELS, -KERNEL_SIZE)를 적용
                * KERNEL은 초기 위치로 되돌림
            * KERNEL_SIZE 루프마다 (0, 0, CONV_STRIDE)를 적용
        '''


        # 프로그램 ID 계산
        pid = tl.program_id(axis=0)
        
        # 각 차원에 대한 블록 수 계산
        num_pid_batch = tl.cdiv(BATCH, BLOCK_SIZE_BATCH)
        num_pid_out_channels = tl.cdiv(OUT_CHANNELS, BLOCK_SIZE_OUT_CHANNELS)
        num_pid_out_length = tl.cdiv(OUT_LENGTH, 1)
        
        # 전체 블록 수
        num_pid_in_group = GROUP_SIZE * num_pid_out_channels * num_pid_out_length
        
        # 그룹 ID 및 그룹 내 위치 계산
        group_id = pid // num_pid_in_group
        first_pid_batch = group_id * GROUP_SIZE
        group_size_batch = min(num_pid_batch - first_pid_batch, GROUP_SIZE)
        
        # 출력 채널과 길이에 대한 프로그램 ID 계산
        pid_batch = first_pid_batch + (pid % group_size_batch)
        pid_out_channels_and_length = (pid % num_pid_in_group) // group_size_batch
        pid_out_channels = pid_out_channels_and_length // num_pid_out_length
        pid_out_length = pid_out_channels_and_length % num_pid_out_length
        
        x_block_pointer = tl.make_block_ptr(
            base= x_pointer,
            shape= (BATCH, IN_CHANNELS, IN_LENGTH),
            strides= (stride_x_batch, stride_x_in_channels, stride_x_in_length),
            offsets= (pid_batch * BLOCK_SIZE_BATCH, 0, pid_out_length),
            block_shape= (BLOCK_SIZE_BATCH, BLOCK_SIZE_IN_CHANNELS, 1), # length dimension is always 1 for BLAS.
            order= (1, 0, 2)
            )

        weights_block_pointer = tl.make_block_ptr(
            base= weights_pointer,
            shape= (OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE),
            strides= (stride_weights_out_channels, stride_weights_in_channels, stride_weights_kernel_size),
            offsets= (pid_out_channels * BLOCK_SIZE_OUT_CHANNELS, 0, 0),
            block_shape= (BLOCK_SIZE_OUT_CHANNELS, BLOCK_SIZE_IN_CHANNELS, 1),  # kernel_size dimension is always 1 for BLAS.
            order= (2, 1, 0)
            )
        
        # Initialize accumulator
        accumulator = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_OUT_CHANNELS, 1), dtype=tl.float32)    # BLOCK_SIZE_OUT_LENGTH is always 1.
        # Convolution loop
        for in_channels_index in range(0, IN_CHANNELS, BLOCK_SIZE_IN_CHANNELS):
            for kernel_size_index in range(0, KERNEL_SIZE, 1):
                x_block = tl.load(x_block_pointer, boundary_check=(0, 1, 2), padding_option='zero').reshape(BLOCK_SIZE_BATCH, BLOCK_SIZE_IN_CHANNELS)
                weights_block = tl.load(weights_block_pointer, boundary_check=(0, 1, 2), padding_option='zero').reshape(BLOCK_SIZE_OUT_CHANNELS, BLOCK_SIZE_IN_CHANNELS)
                
                accumulator += tl.dot(x_block, weights_block.trans(1, 0)).reshape(BLOCK_SIZE_BATCH, BLOCK_SIZE_OUT_CHANNELS, 1)  # tensor.T has a bug
                
                x_block_pointer = tl.advance(x_block_pointer, (0, 0, 1))    # before dilated.
                weights_block_pointer = tl.advance(weights_block_pointer, (0, 0, 1))
            
            x_block_pointer = tl.advance(x_block_pointer, (0, BLOCK_SIZE_IN_CHANNELS, -KERNEL_SIZE))
            weights_block_pointer = tl.advance(weights_block_pointer, (0, BLOCK_SIZE_IN_CHANNELS, -KERNEL_SIZE))

        # Store output
        accumulator = accumulator.to(DTYPE)
        y_block_pointer = tl.make_block_ptr(
            base= y_pointer,
            shape= (BATCH, OUT_CHANNELS, OUT_LENGTH),
            strides= (stride_y_batch, stride_y_out_channels, stride_y_out_length),
            offsets= (pid_batch * BLOCK_SIZE_BATCH, pid_out_channels * BLOCK_SIZE_OUT_CHANNELS, pid_out_length),
            block_shape= (BLOCK_SIZE_BATCH, BLOCK_SIZE_OUT_CHANNELS, 1),
            order= (0, 1, 2)
        )
        tl.store(y_block_pointer, accumulator, boundary_check=(0, 1, 2))

class _Conv1d_without_Bias_Triton_Func(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weights: torch.Tensor,
        *args,
        **kwargs
        ):
        assert x.size(1) == weights.size(1)
        # return triton triton.ops.matmul(x, weights)
        BATCH, IN_CHANNELS, IN_LENGTH = x.size()
        OUT_CHANNELS, _, KERNEL_SIZE = weights.size()
        OUT_LENGTH = (IN_LENGTH - KERNEL_SIZE) + 1
        
        ctx.save_for_backward(x, weights)
        
        y = torch.empty(BATCH, OUT_CHANNELS, OUT_LENGTH, device= x.device, dtype= x.dtype)
        grid = lambda META: (
            triton.cdiv(BATCH, META['BLOCK_SIZE_BATCH']) * 
            triton.cdiv(IN_CHANNELS, META['BLOCK_SIZE_IN_CHANNELS']) * 
            triton.cdiv(OUT_CHANNELS, META['BLOCK_SIZE_OUT_CHANNELS']) *
            triton.cdiv(IN_LENGTH, 1),
            )
        
        _Conv1d_without_Bias_Triton_Kernel.forward[grid](
            x_pointer= x,
            weights_pointer= weights,
            y_pointer= y,  # [Batch, Out]
            
            BATCH= BATCH,
            IN_CHANNELS= IN_CHANNELS,
            IN_LENGTH= IN_LENGTH,
            OUT_CHANNELS= OUT_CHANNELS,
            KERNEL_SIZE= KERNEL_SIZE,
            OUT_LENGTH= OUT_LENGTH,

            stride_x_batch= x.stride(0),
            stride_x_in_channels= x.stride(1),
            stride_x_in_length= x.stride(2),

            stride_weights_out_channels= weights.stride(0),
            stride_weights_in_channels= weights.stride(1),
            stride_weights_kernel_size= weights.stride(2),

            stride_y_batch= y.stride(0),
            stride_y_out_channels= y.stride(1),
            stride_y_out_length= y.stride(2),

            DTYPE= tl.float16 if x.dtype == torch.float16 else tl.float32
            )

        return y
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        y_grad, = grad_outputs
        x, weights = ctx.saved_tensors

        M, K = x.size()
        _, N = weights.size()

        x_grad = torch.empty_like(x)
        weights_grad = torch.empty_like(weights)

        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
        _Linear_without_Bias_Triton_Kernel.backward[grid](
            x_pointer= x, # [Batch, In]
            x_grad_pointer= x_grad, # [Batch, In]
            weights_pointer= weights,    # [In, Out]
            weights_grad_pointer= weights_grad, # [In, Out]
            y_grad_pointer= y_grad,  # [Batch, Out]
            M= M,
            N= N,
            K= K,
            stride_x_m= x.stride(0),
            stride_x_k= x.stride(1),
            stride_weights_k= weights.stride(0),
            stride_weights_n= weights.stride(1),
            stride_y_m= y_grad.stride(0),
            stride_y_n= y_grad.stride(1),
            DTYPE= tl.float16 if x.dtype == torch.float16 else tl.float32
            )

        return x_grad, weights_grad
conv1d_without_bias = _Conv1d_without_Bias_Triton_Func.apply


@triton.autotune(
    configs=get_cuda_autotune_config_matmul_forward(),
    key=['M', 'N', 'K'],
    )
@triton.jit
def _matmul_triton_kernel(
    x1_pointer, # [Batch, In]
    x2_pointer,    # [In, Out]
    y_pointer,  # [Batch, Out]
    M: int,
    N: int,
    K: int,
    stride_x1_m: int,
    stride_x1_k: int,
    stride_x2_k: int,
    stride_x2_n: int,
    stride_y_m: int,
    stride_y_n: int,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DTYPE: tl.constexpr
    ):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + ((pid % group_size_m) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    x1_block_pointer = tl.make_block_ptr(
        base= x1_pointer,
        shape= (M, K),
        strides= (stride_x1_m, stride_x1_k),
        offsets= (pid_m * BLOCK_SIZE_M, 0),
        block_shape= (BLOCK_SIZE_M, BLOCK_SIZE_K),
        order= (1, 0)   # matmul 연산에서 x는 단일 행 내의 여러 열로부터 접근하여 값을 가져옴
        )
    x2_block_pointer = tl.make_block_ptr(
        base= x2_pointer,
        shape= (K, N),
        strides= (stride_x2_k, stride_x2_n),
        offsets= (0, pid_n * BLOCK_SIZE_N),
        block_shape= (BLOCK_SIZE_K, BLOCK_SIZE_N),
        order= (0, 1)   # matmul 연산에서 weight는 단일 열 내의 여러 행으로부터 접근하여 값을 가져옴
        )
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype= tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x1_block = tl.load(x1_block_pointer, boundary_check= (1, 0), padding_option= 'zero')
        x2_block = tl.load(x2_block_pointer, boundary_check= (0, 1), padding_option= 'zero')
        accumulator += tl.dot(x1_block, x2_block)
        x1_block_pointer = tl.advance(x1_block_pointer, (0, BLOCK_SIZE_K))
        x2_block_pointer = tl.advance(x2_block_pointer, (BLOCK_SIZE_K, 0))

    accumulator = accumulator.to(DTYPE)
    y_block_pointer = tl.make_block_ptr(
        base= y_pointer,
        shape= (M, N),
        strides= (stride_y_m, stride_y_n),
        offsets= (pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape= (BLOCK_SIZE_M, BLOCK_SIZE_N),
        order= (1, 0)
        )
    tl.store(y_block_pointer, accumulator, boundary_check= (1, 0))

class _Conv1d_Unfold_without_Bias_Triton_Func(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weights: torch.Tensor,
        *args,
        **kwargs
        ):
        assert x.size(1) == weights.size(1)
        # return triton triton.ops.matmul(x, weights)
        BATCH, IN_CHANNELS, IN_LENGTH = x.size()
        OUT_CHANNELS, _, KERNEL_SIZE = weights.size()
        OUT_LENGTH = (IN_LENGTH - KERNEL_SIZE) + 1
        
        ctx.save_for_backward(x, weights)

        unfolded_x = torch.nn.functional.unfold(
            input= x[:, :, :, None],
            kernel_size= (KERNEL_SIZE, 1),
            dilation= (1, 1),
            stride= (1, 1)
            )
        unfolded_x = unfolded_x.view(BATCH, IN_CHANNELS * KERNEL_SIZE, OUT_LENGTH).mT   # [Batch, Out_Len, In_Ch * Kernel]
        unfolded_x = unfolded_x.reshape(BATCH * OUT_LENGTH, IN_CHANNELS * KERNEL_SIZE) # [Batch * Out_Len, In_Ch * Kernel]

        unfolded_weights = weights.view(OUT_CHANNELS, -1).T  # [In_Ch * Kernel, Out_Ch]

        M, K = unfolded_x.size()
        _, N = unfolded_weights.size()
        unfolded_y = torch.empty(M, N, device= unfolded_x.device, dtype= unfolded_x.dtype)
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
        
        _matmul_triton_kernel[grid](
            x1_pointer= unfolded_x,
            x2_pointer= unfolded_weights,
            y_pointer= unfolded_y,  # [Batch, Out]            
            M= M,
            N= N,
            K= K,
            stride_x1_m= unfolded_x.stride(0),
            stride_x1_k= unfolded_x.stride(1),
            stride_x2_k= unfolded_weights.stride(0),
            stride_x2_n= unfolded_weights.stride(1),
            stride_y_m= unfolded_y.stride(0),
            stride_y_n= unfolded_y.stride(1),
            DTYPE= tl.float16 if x.dtype == torch.float16 else tl.float32
            )        
        y = unfolded_y.view(BATCH, OUT_LENGTH, OUT_CHANNELS).mT

        return y

conv1d_unfold_without_bias = _Conv1d_Unfold_without_Bias_Triton_Func.apply



def validate_without_bias(
    BATCH: int= 16,
    IN_CHANNELS: int= 192,
    IN_LENGTH: int= 173,
    OUT_CHANNELS: int= 384,
    KERNEL_SIZE: int= 3,
    ):
    x_torch = torch.randn(
        BATCH,
        IN_CHANNELS,
        IN_LENGTH,
        device= 'cuda',
        requires_grad= True,
        dtype= torch.float16
        )
    weights_torch = torch.randn(
        OUT_CHANNELS,
        IN_CHANNELS,
        KERNEL_SIZE,
        device= 'cuda',
        requires_grad= True,
        dtype= torch.float16
        )
    y_torch = torch.nn.functional.conv1d(
        input= x_torch,
        weight= weights_torch,
        bias= None,
        stride= 1,
        dilation= 1
        )
    
    x_triton = x_torch.clone().detach()
    weights_triton = weights_torch.clone().detach()
    x_triton.requires_grad = True
    weights_triton.requires_grad = True
    y_triton = conv1d_without_bias(x_triton, weights_triton)

    x_triton_unfold = x_torch.clone().detach()
    weights_triton_unfold = weights_torch.clone().detach()
    x_triton_unfold.requires_grad = True
    weights_triton_unfold.requires_grad = True
    y_triton_unfold = conv1d_unfold_without_bias(x_triton_unfold, weights_triton_unfold)


    difference_torch_triton = (y_torch.data - y_triton.data).abs()
    difference_torch_triton_unfold = (y_torch.data - y_triton_unfold.data).abs()
    print('Forward validate Torch vs Triton: ', torch.allclose(y_torch, y_triton, rtol= 1e-2))
    print(difference_torch_triton.mean(), difference_torch_triton.min(), difference_torch_triton.max())
    print('Forward validate Torch vs Triton Unfold: ', torch.allclose(y_torch, y_triton_unfold, rtol= 1e-2))
    print(difference_torch_triton_unfold.mean(), difference_torch_triton_unfold.min(), difference_torch_triton_unfold.max())
    

@triton.testing.perf_report(triton.testing.Benchmark(
    x_names=['IN_CHANNELS', 'OUT_CHANNELS'],  # Argument names to use as an x-axis for the plot
    x_vals=[128 * i for i in range(2, 10)],  # Different possible values for `x_name`
    line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
    line_vals=['torch', 'triton', 'triton_unfold'],
    line_names=['Torch', 'Triton', 'Triton_Unfold'],
    styles=[('green', '-'), ('blue', '-'), ('red', '-')],
    ylabel='TFLOPS',  # Label name for the y-axis
    plot_name='matmul-performance',
    args={},
    ))
def benchmark_without_bias(IN_CHANNELS, OUT_CHANNELS, provider):
    BATCH = 16
    IN_LENGTH = 3 # 384
    KERNEL_SIZE = 3
    OUT_LENGTH = (IN_LENGTH - KERNEL_SIZE) + 1

    def torch_test():
        x = torch.randn((BATCH, IN_CHANNELS, IN_LENGTH), device='cuda', dtype=torch.float16, requires_grad= True)
        weights = torch.randn((OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE), device='cuda', dtype=torch.float16, requires_grad= True)        
        y = torch.nn.functional.conv1d(x, weights, None, 1, 0, 1)
        
    def triton_test():    
        x = torch.randn((BATCH, IN_CHANNELS, IN_LENGTH), device='cuda', dtype=torch.float16, requires_grad= True)
        weights = torch.randn((OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE), device='cuda', dtype=torch.float16, requires_grad= True)        
        y = conv1d_without_bias(x, weights)

    def triton_unfold_test():    
        x = torch.randn((BATCH, IN_CHANNELS, IN_LENGTH), device='cuda', dtype=torch.float16, requires_grad= True)
        weights = torch.randn((OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE), device='cuda', dtype=torch.float16, requires_grad= True)        
        y = conv1d_unfold_without_bias(x, weights)

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(torch_test, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(triton_test, quantiles=quantiles)
    if provider == 'triton_unfold':
        ms, min_ms, max_ms = triton.testing.do_bench(triton_unfold_test, quantiles=quantiles)
    perf = lambda ms: 2 * OUT_CHANNELS * IN_CHANNELS * KERNEL_SIZE * OUT_LENGTH * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == '__main__':
    validate_without_bias()
    benchmark_without_bias.run(show_plots=True, print_data=True)