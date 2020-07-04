#include"cuckoo_serial.hpp"
#include"cuckoo_cuda.cuh"
#include <stdio.h>
#include <ctime>
#include <ratio>
#include <chrono>
#include <string.h>
uint32_t TABLE_SIZE = 0x1 << 25;
uint32_t DATA_LIMIT = 0x1 << 29;
using namespace std::chrono;

// Generate random data of size items
void gen_random_data(uint32_t*& S, int size)
{
    for (int i = 0; i < size; i++)
    {
        S[i] = rnd() % DATA_LIMIT;
        //std::cout << rnd() << std::endl;   
    }
 
}



int main(int argc, char* argv[])
{
    srand(time(NULL));
    /* Experiment1 */
    
    bool experiment1 = false;
    bool experiment2 = false;
    bool experiment3 = false;
    bool experiment4 = false;
    
    for (int i = 0; i < argc; i++)
    {
        if (strcmp(argv[i],"1") == 0)
            experiment1 = true;
        if (strcmp(argv[i],"2") == 0)
            experiment2 = true;
        if (strcmp(argv[i],"3") == 0)
            experiment3 = true;
        if (strcmp(argv[i],"4") == 0)
            experiment4 = true;
    }

    if (experiment1)
    {   
        std::cout << "-----------------Experiment1 Start-----------------" << std::endl;
        uint32_t* S;
        for(int num_funcs = 3; num_funcs < 5; num_funcs++) // number of hash_funcs 
        {   
            printf("############# %d functions #############\n",num_funcs);
            /* Insert the 10 dataset, each containing 2^i random keys, i=10, 11, ... , 24 */ 
            for (int data_size = 10; data_size < 25; data_size++) 
            {
                std::cout << "%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
                for (int iter = 0; iter < 5; iter++) // 5 iter experiment
                {
                    std::cout << "Datasize:"<< data_size << " Iter:" << iter << std::endl;
            
                    uint size = 0x1 << data_size;
                    S = new uint32_t[size];
                    gen_random_data(S, size);

                    /* Hash class */
                    cuckoo_serial hash_serial = cuckoo_serial(TABLE_SIZE, num_funcs, 4 * ceil(log2(TABLE_SIZE)));
                    cuckoo_cuda hash_cuda = cuckoo_cuda(TABLE_SIZE,num_funcs, 4 * ceil(log2(TABLE_SIZE)));

                    /* Serial insert and get duration */
                    steady_clock::time_point s_start = steady_clock::now();
                    for (int i = 0; i < size; i++)
                        hash_serial.insert(S[i]);
                    //std::cout << hash_serial.a << std::endl;
                    steady_clock::time_point s_end = steady_clock::now();
                    duration<double> serial_time = duration_cast<duration<double>>(s_end - s_start);

                    /* Cuda insert and get duration */
                    steady_clock::time_point c_start = steady_clock::now();
                    hash_cuda.insert(S, size);
                    steady_clock::time_point c_end = steady_clock::now();
                    duration<double> cuda_time = duration_cast<duration<double>>(c_end - c_start);

                    std::cout << "serial_time:" << serial_time.count() << " cuda_time:" << cuda_time.count() << std::endl;
                    //hash_table.show_table();

                    delete[] S;
                }
                std::cout << "%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
            }
        }

        std::cout << "-----------------Experiment1 End-----------------" << std::endl;
    }
    

    /* Experiment2 */
    if(experiment2)
    {
        std::cout << "-----------------Experiment2 Start-----------------" << std::endl;
        for (int num_funcs = 3; num_funcs < 5 ; num_funcs++)
        {
            printf("############# %d functions #############\n",num_funcs);
            for (int iter = 0; iter < 5; iter++)
            {
                std::cout << "[Iter " << iter << "]: " ;
                /* Generate data */
                uint data_size = 0x1 << 24;
                uint32_t *S = new uint32_t[data_size];
                gen_random_data(S, data_size);

                /* Generate data to look up */
                uint32_t *S_lookup[10];
                for (int i = 0; i < 10; i++)
                {   
                    S_lookup[i] = new uint32_t[data_size];
                    int fixed_num = data_size * (10-i)/10;
                    for (int j = 0; j < fixed_num; j++)
                    {
                        int rand_index = rnd() % data_size;
                        S_lookup[i][j] = S[rand_index];
                    }
                    for (int j = fixed_num; j < data_size; j++)
                        S_lookup[i][j] = rnd() % DATA_LIMIT;
                        
                } 
                
                /* Hash class */
                cuckoo_serial hash_serial = cuckoo_serial(TABLE_SIZE,num_funcs,4 * ceil(log2(TABLE_SIZE)));
                cuckoo_cuda hash_cuda = cuckoo_cuda(TABLE_SIZE, num_funcs, 4 * ceil(log2(TABLE_SIZE)));

                /* Insert data */
                for (int i = 0; i < data_size; i++)
                    hash_serial.insert(S[i]);
                hash_cuda.insert(S, data_size);

                /* look up in serial */
                steady_clock::time_point s_start = steady_clock::now();
                for (int i = 0; i < 10; i++)
                {
                    for (int j = 0; j < data_size; j++)
                        hash_serial.look_up(S_lookup[i][j]);
                }
                steady_clock::time_point s_end = steady_clock::now();
                duration<double> serial_time = duration_cast<duration<double>>(s_end - s_start);

                /* Cuda look up */
                steady_clock::time_point c_start = steady_clock::now();
                bool *result = new bool[data_size];
                for (int i = 0; i < 10; i++)
                    hash_cuda.look_up(S_lookup[i], data_size, result);
                steady_clock::time_point c_end = steady_clock::now();
                duration<double> cuda_time = duration_cast<duration<double>>(c_end - c_start);


                std::cout << "serial_time:" << serial_time.count() << " cuda_time:" << cuda_time.count() << std::endl;

                for (int i = 0; i < 10; i++)
                    delete[] S_lookup[i];
                delete[] result;
                delete[] S;
            }
        }
        std::cout << "-----------------Experiment2 End-----------------" << std::endl;
    }


    if(experiment3)
    {
        /* Experiment3 */
        std::cout << "-----------------Experiment3 Start-----------------" << std::endl;
        int data_size = 0x1 << 24;
        uint32_t *S = new uint32_t[data_size];
        gen_random_data(S, data_size);

        for (int num_funcs = 3; num_funcs < 5 ; num_funcs++)
        {
            printf("############# %d functions #############\n",num_funcs);
            // From 1.1 to 2.0
            for (float scale = 1.1; scale < 2.0; scale+=0.1)
            {
                std::cout << "%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
                for (int iter = 0; iter < 5; iter++)
                {
                    std::cout << "[scale:" << scale << " iter:" << iter << "]" << std::endl;
                    uint hash_size = ceil(scale * data_size);
                    cuckoo_serial hash_serial = cuckoo_serial(hash_size,num_funcs,10 * ceil(log2(hash_size)));
                    cuckoo_cuda hash_cuda = cuckoo_cuda(hash_size, num_funcs,10 * ceil(log2(hash_size)));

                    /* Serial insert */
                    steady_clock::time_point s_start = steady_clock::now();
                    for (int i = 0; i < data_size; i++)
                        hash_serial.insert(S[i]);
                    steady_clock::time_point s_end = steady_clock::now();
                    duration<double> serial_time = duration_cast<duration<double>>(s_end - s_start);

                    /* Cuda insert */
                    steady_clock::time_point c_start = steady_clock::now();
                    hash_cuda.insert(S, data_size);
                    steady_clock::time_point c_end = steady_clock::now();
                    duration<double> cuda_time = duration_cast<duration<double>>(c_end - c_start);

                    std::cout << "serial_time:" << serial_time.count() << " cuda_time:" << cuda_time.count() << std::endl;
                }
            }

            // From 1.01 to 1.05
            for (float scale = 1.01; scale < 1.05; scale+=0.01)
            {
                std::cout << "%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
                for (int iter = 0; iter < 5; iter++)
                {
                    std::cout << "[scale:" << scale << " iter:" << iter << "]" << std::endl;
                    uint hash_size = ceil(scale * data_size);
                    cuckoo_serial hash_serial = cuckoo_serial(hash_size, num_funcs, 10 * ceil(log2(hash_size)));
                    cuckoo_cuda hash_cuda = cuckoo_cuda(hash_size, num_funcs, 10 * ceil(log2(hash_size)));

                    /* Serial insert */
                    steady_clock::time_point s_start = steady_clock::now();
                    for (int i = 0; i < data_size; i++)
                        hash_serial.insert(S[i]);
                    steady_clock::time_point s_end = steady_clock::now();
                    duration<double> serial_time = duration_cast<duration<double>>(s_end - s_start);

                    /* Cuda insert */
                    steady_clock::time_point c_start = steady_clock::now();
                    hash_cuda.insert(S, data_size);
                    steady_clock::time_point c_end = steady_clock::now();
                    duration<double> cuda_time = duration_cast<duration<double>>(c_end - c_start);

                    std::cout << "serial_time:" << serial_time.count() << " cuda_time:" << cuda_time.count() << std::endl;
                }
            }
        }

        delete[] S;
    }


    if(experiment4)
    {
        /* Experiment4 */
        std::cout << "-----------------Experiment4 Start-----------------" << std::endl;

        int data_size = 0x1 << 24;
        uint32_t *S = new uint32_t[data_size];
        gen_random_data(S, data_size);
        int hash_size = 1.4 * data_size;
        
        for (int bound = 1; bound < 10; bound++)
        {
            std::cout << "[Bound " << bound << "]";
            cuckoo_serial hash_serial = cuckoo_serial(hash_size, 2, bound * ceil(log2(hash_size)) );
            cuckoo_cuda hash_cuda = cuckoo_cuda(hash_size, 2, bound * ceil(log2(hash_size)));

            /* Serial insert */
            steady_clock::time_point s_start = steady_clock::now();
            for (int i = 0; i < data_size; i++)
                hash_serial.insert(S[i]);
            steady_clock::time_point s_end = steady_clock::now();
            duration<double> serial_time = duration_cast<duration<double>>(s_end - s_start);
    
            /* Cuda insert */
            steady_clock::time_point c_start = steady_clock::now();
            hash_cuda.insert(S, data_size);
            steady_clock::time_point c_end = steady_clock::now();
            duration<double> cuda_time = duration_cast<duration<double>>(c_end - c_start);
    
            std::cout << "serial_time:" << serial_time.count() << " cuda_time:" << cuda_time.count() << std::endl;
        }


    }   


    return 0;
}