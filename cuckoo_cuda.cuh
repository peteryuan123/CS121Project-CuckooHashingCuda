#ifndef CUCKOO_CUDA
#define CUCKOO_CUDA

#include<iostream>
#include"util.h"


__device__ uint32_t 
compress_data(uint32_t key, uint func_index)
{
    return ( key << 2 ) | func_index;
}

__device__ uint32_t 
extract_key(uint32_t compressed_data)
{
    return  compressed_data >> 2;
}

__device__ uint32_t 
extract_func_idx(uint32_t compressed_data)
{
    return compressed_data & 0b11;
}

__device__ uint 
hash_func(uint32_t key, para parameter, uint32_t prime, uint size)
{
    //return ((key ^ parameter.a) >> parameter.b) % _size;
    return ((key * parameter.a + parameter.b) % prime) % size ;
}


class cuckoo_cuda
{

    
private:
    const uint _size;         // Size of the hash table
    uint _num_func;           // Number of hash functions
    uint32_t* _table;         // Data
    para *_hash_func_param; // Store all the divisors. As the form of hash function we use is:  ((a*key+b) mod prime) mod size

    uint _eviction_bound;     // When the chain's length reaches evication bound, rebuid the hash table
    uint32_t _prime = 35003489;

private:

    void rebuild_table(uint32_t key);

    /*  @brief General form of hash function
              It is easy to re-generate new hash functions when we reach eviction bound 
        @return
                index of the key
    */
    uint hash_func(uint32_t key, para parameter)
    {
        //return ((key ^ parameter.a) >> parameter.b) % _size;
        return ((key * parameter.a + parameter.b) % _prime) % _size ;
    }

    // Re-generate new divisior
    void gen_hash_divisor()
    {
        for (uint i = 0; i < _num_func; i++)
        {
            _hash_func_param[i].a = rnd() % _prime;
            _hash_func_param[i].b = rnd() % _prime;
            //std::cout << _hash_func_param[i].a << " " << _hash_func_param[i].b << std::endl;
        }
    }

 

public:

    cuckoo_cuda(const uint size,uint num_func);
    ~cuckoo_cuda();

    void insert(uint32_t key);
    bool remove(uint32_t key);
    bool look_up(uint32_t key);
    void show_table();
};

cuckoo_cuda::cuckoo_cuda(const uint size, uint num_func): _size(size), _num_func(num_func)
{
    _hash_func_param = new para[num_func];
    table = new uint32_t[size];
    gen_hash_divisor();
    _eviction_bound = 4 * ceil(log2(size));
}

__global__ void
InsertKernel(uint32_t *keys, int n, uint32_t* &table, 
            para *hash_func_param, uint eviction_bound,
            uint32_t prime, uint size, uint num_func )
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        uint32_t m_key = keys[idx];
        uint hash_index = 0;

        uint chain_count = 0;
        while (chain_count < eviction_bound)
        {
            para param = hash_func_param[hash_index];
            uint data_index = hash_func(m_key, param, prime, size);

            uint32_t compressed_data =  compress_data(m_key, hash_index);
            uint32_t old = atomicExch(&table[data_index], compressed_data);
            if (old == 0)
                return;
            else
            {
                m_key = extract_key(compressed_data);
                hash_index = (extract_func_idx(compressed_data) + 1) % num_func;
                chain_count++;
            }
        }
    }


}

__global__ void
LookupKernel(uint32_t *keys, int n, uint32_t* &table, 
            para *hash_func_param, uint32_t prime,
             uint size, uint num_func,  bool *result)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        uint32_t m_key = keys[idx];
        for (int i = 0; i < num_func; i++)
        {
            para param = hash_func_param[i];
            uint data_index = hash_func(m_key, param, prime, size);
            uint32_t cur_key = extract_key(table[data_index]);
            if (m_key == cur_key)
            {
                result[idx] = true;
                return;
            }
        }

        result[idx] = false;
    }
}

__global__ void
DeleteKernel(uint32_t *keys, int n, uint32_t* &table, 
    para *hash_func_param, uint32_t prime,
     uint size, uint num_func,  bool *result)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n)
    {
        uint32_t m_key = keys[idx];
        for (int i = 0; i < num_func; i++)
        {
            param = hash_func_param[i];
            uint data_index = hash_func(m_key, param, prime,size);
            uint32_t cur_key = extract_key(table[data_index]);
            if (m_key == cur_key)
                atomicExch(&table[data_index], 0);
        }
    }

}

#endif 