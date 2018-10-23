# CUDA_Notebook #
****
## 并行通信模式 ##
1. 映射(map)：一对一的内存映射模式
2. 聚合(gatter)：多个输入对应一个输出
3. 分散(scatter)：一个输入对应多个输出
4. 模板(stencil)：以固定模式读取相邻内存的数值
5. 转置(transpose)：一对一 类似于矩阵转置
6. 压缩(reduce)
7. 重排(scan/sort)
## GPU硬件模式 ##
- thread
> 线程  每个线程用来处理一个简单运算

- thread block
> 线程块  每个线程块包含若干个线程

- grid
> 网格 由若干个线程块组成 一个网格用来解决一个kernel

- Kernel
> 相当于一个函数function 用来解决一个问题

- Stream Processor(流处理器)
> 每个gpu有若干个sm，每个sm独立工作  sm包括若干个simple processor和memory，simple processor用来处理线程。

![GPU模型](https://github.com/zhuangzhuangsun/CUDA/blob/master/gpu.jpg)
## CUDA内存模型 ##
## CUDA编程模型 ##
1. 对线程块分配到哪里运行不做保证，不保证何时运行完
2. 所有在同一个线程块上的线程必然会在同一时间同时运行在同一个SM上
3. 同一个内核的所有线程块必然会全部完成后才会运算下一个内核

同步性synchronisation和屏障barrier

barrier：用来控制多个线程停止与等待，当所有线程都达到了屏障点，程序才继续运行


同步性：不同线程在共享和全局内存中读写数据要有先后控制

#### cuda编程流程 ####
1. CPU给GPU分配空间(cudamalloc)
2. CPU给GPU复制数据(cudamemcpy)
3. 加载kernel给GPU
4. CPU把GPU计算结果复制回来(cudamemcpy)
## 例程 ##
    #include <stdio.h>
    __global__ void square(float* d_out,float* d_in){
      int idx = threadIdx.x;
      float f = d_in[idx];
      d_out[idx] = f * f;
    }
    
    int main(int argc,char** argv){
      const int ARRAY_SIZE = 8;
      const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
    
      // generate the input array on the host
      float h_in[ARRAY_SIZE];
      for(int i=0;i<ARRAY_SIZE;i++){
    h_in[i] = float(i);
      }
      float h_out[ARRAY_SIZE];
    
      // declare GPU memory pointers
      float* d_in;
      float* d_out;
    
      // allocate GPU memory
      cudaMalloc((void**) &d_in,ARRAY_BYTES);
      cudaMalloc((void**) &d_out,ARRAY_BYTES);
    
      // transfer the array to GPU
      cudaMemcpy(d_in,h_in,ARRAY_BYTES,cudaMemcpyHostToDevice);
    
      // launch the kernel
      square<<<1,ARRAY_SIZE>>>(d_out,d_in);
    
      // copy back the result array to the GPU
      cudaMemcpy(h_out,d_out,ARRAY_BYTES,cudaMemcpyDeviceToHost);
    
      // print out the resulting array
      for(int i=0;i<ARRAY_SIZE;i++){
    printf("%f",h_out[i]);
    printf(((i%4) != 3) ? "\t" : "\n");
      }
    
      // free GPU memory allocation
      cudaFree(d_in);
      cudaFree(d_out);
    
      return 0;
    
    
    }

