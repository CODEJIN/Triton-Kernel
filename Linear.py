import torch
import triton
import triton.ops
import triton.language as tl

def get_cuda_autotune_config_forward():
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

class _Linear_without_Bias_Triton_Kernel:
    @staticmethod
    @triton.autotune(
        configs=get_cuda_autotune_config_forward(),
        key=['M', 'N', 'K'],
        )
    @triton.jit
    def forward(
        x_pointer, # [Batch, In]
        weights_pointer,    # [In, Out]
        y_pointer,  # [Batch, Out]
        M: int,
        N: int,
        K: int,
        stride_x_m: int,
        stride_x_k: int,
        stride_weights_k: int,
        stride_weights_n: int,
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

        x_block_pointer = tl.make_block_ptr(
            base= x_pointer,
            shape= (M, K),
            strides= (stride_x_m, stride_x_k),
            offsets= (pid_m * BLOCK_SIZE_M, 0),
            block_shape= (BLOCK_SIZE_M, BLOCK_SIZE_K),
            order= (1, 0)   # matmul 연산에서 x는 단일 행 내의 여러 열로부터 접근하여 값을 가져옴
            )
        weights_block_pointer = tl.make_block_ptr(
            base= weights_pointer,
            shape= (K, N),
            strides= (stride_weights_k, stride_weights_n),
            offsets= (0, pid_n * BLOCK_SIZE_N),
            block_shape= (BLOCK_SIZE_K, BLOCK_SIZE_N),
            order= (0, 1)   # matmul 연산에서 weight는 단일 열 내의 여러 행으로부터 접근하여 값을 가져옴
            )
        
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype= tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            x_block = tl.load(x_block_pointer, boundary_check= (1, 0), padding_option= 'zero')
            weights_block = tl.load(weights_block_pointer, boundary_check= (0, 1), padding_option= 'zero')
            accumulator += tl.dot(x_block, weights_block)
            x_block_pointer = tl.advance(x_block_pointer, (0, BLOCK_SIZE_K))
            weights_block_pointer = tl.advance(weights_block_pointer, (BLOCK_SIZE_K, 0))

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

    @staticmethod
    @triton.autotune(
        configs=get_cuda_autotune_config_forward(),
        key=['M', 'N', 'K'],
        )
    @triton.jit    
    def backward(
        x_pointer, # [Batch, In]
        x_grad_pointer,
        weights_pointer,    # [In, Out]
        weights_grad_pointer,
        y_grad_pointer,  # [Batch, Out]
        M: int,
        N: int,
        K: int,
        stride_x_m: int,
        stride_x_k: int,
        stride_weights_k: int,
        stride_weights_n: int,
        stride_y_m: int,
        stride_y_n: int,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        DTYPE: tl.constexpr,
        ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
        num_pid_in_group = GROUP_SIZE * num_pid_k
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE)
        pid_m = first_pid_m + ((pid % group_size_m) % group_size_m)
        pid_k = (pid % num_pid_in_group) // group_size_m

        y_grad_block_pointer = tl.make_block_ptr(
            base= y_grad_pointer,
            shape= (M, N),
            strides= (stride_y_m, stride_y_n),
            offsets= (pid_m * BLOCK_SIZE_M, 0),
            block_shape= (BLOCK_SIZE_M, BLOCK_SIZE_N),
            order= (1, 0)
            )
        weights_block_pointer = tl.make_block_ptr(
            base= weights_pointer,
            shape= (K, N),
            strides= (stride_weights_k, stride_weights_n),
            offsets= (pid_k * BLOCK_SIZE_K, 0),
            block_shape= (BLOCK_SIZE_K, BLOCK_SIZE_N),
            order= (1, 0)   # weight도 K 기준으로 진행
            )

        accumulator_x_grad = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype= tl.float32)
        for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
            y_grad_block = tl.load(y_grad_block_pointer, boundary_check= (1, 0), padding_option= 'zero')
            weights_block = tl.load(weights_block_pointer, boundary_check= (1, 0), padding_option= 'zero')
            accumulator_x_grad += tl.dot(y_grad_block, weights_block.T)
            y_grad_block_pointer = tl.advance(y_grad_block_pointer, (0, BLOCK_SIZE_N))
            weights_block_pointer = tl.advance(weights_block_pointer, (0, BLOCK_SIZE_N))

        accumulator_x_grad = accumulator_x_grad.to(DTYPE)
        x_grad_block_pointer = tl.make_block_ptr(
            base= x_grad_pointer,
            shape= (M, K),
            strides= (stride_x_m, stride_x_k),
            offsets= (pid_m * BLOCK_SIZE_M, pid_k * BLOCK_SIZE_K),
            block_shape= (BLOCK_SIZE_M, BLOCK_SIZE_K),
            order= (1, 0)
            )
        tl.store(x_grad_block_pointer, accumulator_x_grad, boundary_check= (1, 0))

        num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_k = group_id * GROUP_SIZE
        group_size_k = min(num_pid_k - first_pid_k, GROUP_SIZE)
        pid_k = first_pid_k + ((pid % group_size_k) % group_size_k)
        pid_n = (pid % num_pid_in_group) // group_size_k

        x_block_pointer = tl.make_block_ptr(
            base= x_pointer,
            shape= (M, K),
            strides= (stride_x_m, stride_x_k),
            offsets= (0, pid_k * BLOCK_SIZE_K),
            block_shape= (BLOCK_SIZE_M, BLOCK_SIZE_K),
            order= (0, 1)   
            )
        y_grad_block_pointer = tl.make_block_ptr(
            base= y_grad_pointer,
            shape= (M, N),
            strides= (stride_y_m, stride_y_n),
            offsets= (0, pid_n * BLOCK_SIZE_N),
            block_shape= (BLOCK_SIZE_M, BLOCK_SIZE_N),
            order= (0, 1)
            )

        accumulator_weights_grad = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype= tl.float32)
        for m in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
            x_block = tl.load(x_block_pointer, boundary_check= (0, 1), padding_option= 'zero')
            y_grad_block = tl.load(y_grad_block_pointer, boundary_check= (0, 1), padding_option= 'zero')
            accumulator_weights_grad += tl.dot(x_block.T, y_grad_block)
            x_block_pointer = tl.advance(x_block_pointer, (BLOCK_SIZE_M, 0))
            y_grad_block_pointer = tl.advance(y_grad_block_pointer, (BLOCK_SIZE_M, 0))

        accumulator_weights_grad = accumulator_weights_grad.to(DTYPE)
        weights_grad_block_pointer = tl.make_block_ptr(
            base= weights_grad_pointer,
            shape= (K, N),
            strides= (stride_weights_k, stride_weights_n),
            offsets= (pid_k * BLOCK_SIZE_K, pid_n * BLOCK_SIZE_N),
            block_shape= (BLOCK_SIZE_K, BLOCK_SIZE_N),
            order= (0, 1)
            )
        tl.store(weights_grad_block_pointer, accumulator_weights_grad, boundary_check= (0, 1))

class _Linear_with_Bias_Triton_Kernel:
    @staticmethod
    @triton.autotune(
        configs=get_cuda_autotune_config_forward(),
        key=['M', 'N', 'K'],
        )
    @triton.jit
    def forward(
        x_pointer, # [Batch, In]
        weights_pointer,    # [In, Out]
        bias_pointer,   # [Out]
        y_pointer,  # [Batch, Out]
        M: int,
        N: int,
        K: int,
        stride_x_m: int,
        stride_x_k: int,
        stride_weights_k: int,
        stride_weights_n: int,
        stride_bias_n: int,
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

        x_block_pointer = tl.make_block_ptr(
            base= x_pointer,
            shape= (M, K),
            strides= (stride_x_m, stride_x_k),
            offsets= (pid_m * BLOCK_SIZE_M, 0),
            block_shape= (BLOCK_SIZE_M, BLOCK_SIZE_K),
            order= (1, 0)   # matmul 연산에서 x는 단일 행 내의 여러 열로부터 접근하여 값을 가져옴
            )
        weights_block_pointer = tl.make_block_ptr(
            base= weights_pointer,
            shape= (K, N),
            strides= (stride_weights_k, stride_weights_n),
            offsets= (0, pid_n * BLOCK_SIZE_N),
            block_shape= (BLOCK_SIZE_K, BLOCK_SIZE_N),
            order= (0, 1)   # matmul 연산에서 weight는 단일 열 내의 여러 행으로부터 접근하여 값을 가져옴
            )
        bias_block_pointer = tl.make_block_ptr(
            base= bias_pointer,
            shape= (N, ),
            strides= (stride_bias_n,),
            offsets= (pid_n * BLOCK_SIZE_N, ),
            block_shape= (BLOCK_SIZE_N, ),
            order= (0,)
            )
        
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype= tl.float32)        
        bias_block = tl.load(bias_block_pointer, boundary_check= (0, ), padding_option= 'zero')
        accumulator += bias_block
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            x_block = tl.load(x_block_pointer, boundary_check= (1, 0), padding_option= 'zero')
            weights_block = tl.load(weights_block_pointer, boundary_check= (0, 1), padding_option= 'zero')            
            accumulator += tl.dot(x_block, weights_block)
            x_block_pointer = tl.advance(x_block_pointer, (0, BLOCK_SIZE_K))
            weights_block_pointer = tl.advance(weights_block_pointer, (BLOCK_SIZE_K, 0))            

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

    @staticmethod
    @triton.autotune(
        configs=get_cuda_autotune_config_forward(),
        key=['M', 'N', 'K'],
        )
    @triton.jit    
    def backward(
        x_pointer, # [Batch, In]
        x_grad_pointer,
        weights_pointer,    # [In, Out]
        weights_grad_pointer,
        y_grad_pointer,  # [Batch, Out]
        M: int,
        N: int,
        K: int,
        stride_x_m: int,
        stride_x_k: int,
        stride_weights_k: int,
        stride_weights_n: int,
        stride_y_m: int,
        stride_y_n: int,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        DTYPE: tl.constexpr,
        ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
        num_pid_in_group = GROUP_SIZE * num_pid_k
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE)
        pid_m = first_pid_m + ((pid % group_size_m) % group_size_m)
        pid_k = (pid % num_pid_in_group) // group_size_m

        y_grad_block_pointer = tl.make_block_ptr(
            base= y_grad_pointer,
            shape= (M, N),
            strides= (stride_y_m, stride_y_n),
            offsets= (pid_m * BLOCK_SIZE_M, 0),
            block_shape= (BLOCK_SIZE_M, BLOCK_SIZE_N),
            order= (1, 0)
            )
        weights_block_pointer = tl.make_block_ptr(
            base= weights_pointer,
            shape= (K, N),
            strides= (stride_weights_k, stride_weights_n),
            offsets= (pid_k * BLOCK_SIZE_K, 0),
            block_shape= (BLOCK_SIZE_K, BLOCK_SIZE_N),
            order= (1, 0)   # weight도 K 기준으로 진행
            )

        accumulator_x_grad = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype= tl.float32)
        for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
            y_grad_block = tl.load(y_grad_block_pointer, boundary_check= (1, 0), padding_option= 'zero')
            weights_block = tl.load(weights_block_pointer, boundary_check= (1, 0), padding_option= 'zero')
            accumulator_x_grad += tl.dot(y_grad_block, weights_block.T)
            y_grad_block_pointer = tl.advance(y_grad_block_pointer, (0, BLOCK_SIZE_N))
            weights_block_pointer = tl.advance(weights_block_pointer, (0, BLOCK_SIZE_N))

        accumulator_x_grad = accumulator_x_grad.to(DTYPE)
        x_grad_block_pointer = tl.make_block_ptr(
            base= x_grad_pointer,
            shape= (M, K),
            strides= (stride_x_m, stride_x_k),
            offsets= (pid_m * BLOCK_SIZE_M, pid_k * BLOCK_SIZE_K),
            block_shape= (BLOCK_SIZE_M, BLOCK_SIZE_K),
            order= (1, 0)
            )
        tl.store(x_grad_block_pointer, accumulator_x_grad, boundary_check= (1, 0))

        num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_k = group_id * GROUP_SIZE
        group_size_k = min(num_pid_k - first_pid_k, GROUP_SIZE)
        pid_k = first_pid_k + ((pid % group_size_k) % group_size_k)
        pid_n = (pid % num_pid_in_group) // group_size_k

        x_block_pointer = tl.make_block_ptr(
            base= x_pointer,
            shape= (M, K),
            strides= (stride_x_m, stride_x_k),
            offsets= (0, pid_k * BLOCK_SIZE_K),
            block_shape= (BLOCK_SIZE_M, BLOCK_SIZE_K),
            order= (0, 1)   
            )
        y_grad_block_pointer = tl.make_block_ptr(
            base= y_grad_pointer,
            shape= (M, N),
            strides= (stride_y_m, stride_y_n),
            offsets= (0, pid_n * BLOCK_SIZE_N),
            block_shape= (BLOCK_SIZE_M, BLOCK_SIZE_N),
            order= (0, 1)
            )

        accumulator_weights_grad = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype= tl.float32)
        for m in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
            x_block = tl.load(x_block_pointer, boundary_check= (0, 1), padding_option= 'zero')
            y_grad_block = tl.load(y_grad_block_pointer, boundary_check= (0, 1), padding_option= 'zero')
            accumulator_weights_grad += tl.dot(x_block.T, y_grad_block)
            x_block_pointer = tl.advance(x_block_pointer, (BLOCK_SIZE_M, 0))
            y_grad_block_pointer = tl.advance(y_grad_block_pointer, (BLOCK_SIZE_M, 0))

        accumulator_weights_grad = accumulator_weights_grad.to(DTYPE)
        weights_grad_block_pointer = tl.make_block_ptr(
            base= weights_grad_pointer,
            shape= (K, N),
            strides= (stride_weights_k, stride_weights_n),
            offsets= (pid_k * BLOCK_SIZE_K, pid_n * BLOCK_SIZE_N),
            block_shape= (BLOCK_SIZE_K, BLOCK_SIZE_N),
            order= (0, 1)
            )
        tl.store(weights_grad_block_pointer, accumulator_weights_grad, boundary_check= (0, 1))




class _Linear_without_Bias_Triton_Func(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weights: torch.Tensor,
        *args,
        **kwargs
        ):
        assert x.size(1) == weights.size(0)        
        # return triton triton.ops.matmul(x, weights)
        M, K = x.size()
        _, N = weights.size()
        ctx.save_for_backward(x, weights)
        
        y = torch.empty(M, N, device= x.device, dtype= x.dtype)
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
        
        _Linear_without_Bias_Triton_Kernel.forward[grid](
            x_pointer= x,
            weights_pointer= weights,
            y_pointer= y,  # [Batch, Out]
            M= M,
            N= N,
            K= K,
            stride_x_m= x.stride(0),
            stride_x_k= x.stride(1),
            stride_weights_k= weights.stride(0),
            stride_weights_n= weights.stride(1),
            stride_y_m= y.stride(0),
            stride_y_n= y.stride(1),
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

class _Linear_with_Bias_Triton_Func(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weights: torch.Tensor,
        bias: torch.Tensor,
        *args,
        **kwargs
        ):
        assert x.size(1) == weights.size(0)
        assert weights.size(1) == bias.size(0)
        # return triton triton.ops.matmul(x, weights)
        M, K = x.size()
        _, N = weights.size()
        ctx.save_for_backward(x, weights, bias)
        
        y = torch.empty(M, N, device= x.device, dtype= x.dtype)
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
        
        _Linear_with_Bias_Triton_Kernel.forward[grid](
            x_pointer= x,
            weights_pointer= weights,
            bias_pointer= bias,
            y_pointer= y,  # [Batch, Out]
            M= M,
            N= N,
            K= K,
            stride_x_m= x.stride(0),
            stride_x_k= x.stride(1),
            stride_weights_k= weights.stride(0),
            stride_weights_n= weights.stride(1),
            stride_bias_n= bias.stride(0),
            stride_y_m= y.stride(0),
            stride_y_n= y.stride(1),
            DTYPE= tl.float16 if x.dtype == torch.float16 else tl.float32
            )

        return y
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        y_grad, = grad_outputs
        x, weights, _ = ctx.saved_tensors   # bias does not need.

        M, K = x.size()
        _, N = weights.size()

        x_grad = torch.empty_like(x)
        weights_grad = torch.empty_like(weights)
        bias_grad = y_grad

        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
        _Linear_with_Bias_Triton_Kernel.backward[grid](
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

        return x_grad, weights_grad, bias_grad

linear_without_bias = _Linear_without_Bias_Triton_Func.apply
linear_with_bias = _Linear_with_Bias_Triton_Func.apply

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias: bool= True):
        super(Linear, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_features))

    def forward(self, x: torch.Tensor):
        if hasattr(self, 'bias'):
            return linear_with_bias(x, self.weight, self.bias)
        return linear_without_bias(x, self.weight)

def validate_without_bias(M: int= 1024, N: int= 1536, K: int= 768):
    x_torch = torch.randn(M, K, device= 'cuda', dtype= torch.float16, requires_grad= True)
    weights_torch = torch.randn(K, N, device= 'cuda', dtype= torch.float16, requires_grad= True)
    y_torch = x_torch @ weights_torch

    x_triton = x_torch.clone().detach()
    weights_triton = weights_torch.clone().detach()
    x_triton.requires_grad = True
    weights_triton.requires_grad = True
    y_triton = linear_without_bias(x_triton, weights_triton)
    print('Forward validate: ', torch.allclose(y_torch, y_triton, rtol= 1e-2))

    y_torch_grad = torch.randn_like(y_torch)
    y_torch.backward(y_torch_grad)

    y_triton_grad = y_torch_grad.clone().detach()
    y_triton.backward(y_triton_grad)

    print('Backward validate x: ', torch.allclose(x_torch.grad, x_triton.grad, rtol= 1e-2))
    print('Backward validate weights: ', torch.allclose(weights_torch.grad, weights_triton.grad, rtol= 1e-2))

def validate_with_bias(M: int= 1024, N: int= 1536, K: int= 768):
    x_torch = torch.randn(M, K, device= 'cuda', dtype= torch.float16, requires_grad= True)
    weights_torch = torch.randn(K, N, device= 'cuda', dtype= torch.float16, requires_grad= True)
    bias_torch = torch.randn(N, device= 'cuda', dtype= torch.float16, requires_grad= True)
    y_torch = x_torch @ weights_torch + bias_torch

    x_triton = x_torch.clone().detach()
    weights_triton = weights_torch.clone().detach()
    bias_triton = bias_torch.clone().detach()
    x_triton.requires_grad = True
    weights_triton.requires_grad = True
    bias_triton.requires_grad = True
    y_triton = linear_with_bias(x_triton, weights_triton, bias_triton)
    print('Forward validate: ', torch.allclose(y_torch, y_triton, rtol= 1e-2))

    y_torch_grad = torch.randn_like(y_torch)
    y_torch.backward(y_torch_grad)

    y_triton_grad = y_torch_grad.clone().detach()
    y_triton.backward(y_triton_grad)

    print('Backward validate x: ', torch.allclose(x_torch.grad, x_triton.grad, rtol= 1e-2))
    print('Backward validate weights: ', torch.allclose(weights_torch.grad, weights_triton.grad, rtol= 1e-2))
    print('Backward validate bias: ', torch.allclose(bias_torch.grad, bias_triton.grad, rtol= 1e-2))

@triton.testing.perf_report(triton.testing.Benchmark(
    x_names=['M', 'N', 'K'],  # Argument names to use as an x-axis for the plot
    x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
    line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
    line_vals=['torch', 'triton'],
    line_names=['Torch', 'Triton'],
    styles=[('green', '-'), ('blue', '-')],
    ylabel='TFLOPS',  # Label name for the y-axis
    plot_name='matmul-performance',
    args={},
    ))
def benchmark_without_bias(M, N, K, provider):
    def torch_test():    
        a = torch.randn((M, K), device='cuda', dtype=torch.float16, requires_grad= True)
        b = torch.randn((K, N), device='cuda', dtype=torch.float16, requires_grad= True)        
        c = a @ b
        c_grad = torch.randn_like(c)
        c.backward(c_grad)

    def triton_test():    
        a = torch.randn((M, K), device='cuda', dtype=torch.float16, requires_grad= True)
        b = torch.randn((K, N), device='cuda', dtype=torch.float16, requires_grad= True)
        c = linear_without_bias(a, b)
        c_grad = torch.randn_like(c)
        c.backward(c_grad)



    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(torch_test, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(triton_test, quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


@triton.testing.perf_report(triton.testing.Benchmark(
    x_names=['M', 'N', 'K'],  # Argument names to use as an x-axis for the plot
    x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
    line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
    line_vals=['torch', 'triton'],
    line_names=['Torch', 'Triton'],
    styles=[('green', '-'), ('blue', '-')],
    ylabel='TFLOPS',  # Label name for the y-axis
    plot_name='matmul-performance',
    args={},
    ))
def benchmark_with_bias(M, N, K, provider):
    def torch_test():
        x = torch.randn((M, K), device='cuda', dtype=torch.float16, requires_grad= True)
        weights = torch.randn((K, N), device='cuda', dtype=torch.float16, requires_grad= True)
        bias = torch.randn((K, N), device='cuda', dtype=torch.float16, requires_grad= True)
        y = x @ weights + bias
        y_grad = torch.randn_like(y)
        y.backward(y_grad)

    def triton_test():
        x = torch.randn((M, K), device='cuda', dtype=torch.float16, requires_grad= True)
        weights = torch.randn((K, N), device='cuda', dtype=torch.float16, requires_grad= True)
        bias = torch.randn((K, N), device='cuda', dtype=torch.float16, requires_grad= True)
        y = linear_with_bias(x, weights, bias)
        y_grad = torch.randn_like(bias)
        y.backward(y_grad)

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(torch_test, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(triton_test, quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)



if __name__ == '__main__':
    # validate_without_bias()
    # benchmark_without_bias.run(show_plots=True, print_data=True)
    validate_with_bias()
    benchmark_with_bias.run(show_plots=True, print_data=True)
    


