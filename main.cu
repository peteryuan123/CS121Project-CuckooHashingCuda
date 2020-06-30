#include"cuckoo_serial.hpp"
#include"cuckoo_cuda.cuh"
#include <map>
uint32_t TABLE_SIZE = 0x1 << 25;
uint32_t DATA_LIMIT = 0x1 << 29;

void gen_random_data(uint32_t*& S, int size)
{

    std::map<uint32_t, bool> val_map;
    for (int i = 0; i < size; i++)
    {
        S[i] = rnd() % DATA_LIMIT  ;
        //std::cout <<  S[i] << std::endl;
    }
    
}



int main()
{
    srand(time(NULL));
    /* Experiment1 */
    {

        std::cout << "-----------------Experiment1-----------------" << std::endl;
        uint32_t* S;
        for (int num_funcs = 2; num_funcs < 4; num_funcs++)
        {
            /* Insert the 10 dataset, each containing 2^i random keys, i=10, 11, ... , 24 */
            for (int data_size = 24; data_size < 25; data_size++)
            {
                std::cout << data_size <<std::endl;
        
                uint size = 0x1 << data_size;
                S = new uint32_t[size];
                gen_random_data(S, size);
                //std::cout << S[0] << std::endl;
                cuckoo_serial hash_table = cuckoo_serial(TABLE_SIZE, num_funcs);
                for (int i = 0; i < size; i++)
                    hash_table.insert(S[i]);
                
                hash_table.show_table();

                delete[] S;
            }

        }

        
    }
    

    /* Experiment2 */
    {
        std::cout << "-----------------Experiment2-----------------" << std::endl;
        uint data_size = 0x1 << 24;
        uint32_t *S = new uint32_t[data_size];
        gen_random_data(S, data_size);

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
        cuckoo_serial hash_table = cuckoo_serial(TABLE_SIZE,2);

        /*for (int i = 10; i < 25; i++)
        {
            uint size = 0x1 << i;
            std::cout << i << std::endl;
            for (int j = 0; j < size; j++)
                std::cout << hash_table.look_up(S[i-10][j]);
        }*/

        delete[] S;
    }
    /* Experiment3 */
    std::cout << "-----------------Experiment3-----------------" << std::endl;


    /* Experiment4 */
    std::cout << "-----------------Experiment4-----------------" << std::endl;

    return 0;
}