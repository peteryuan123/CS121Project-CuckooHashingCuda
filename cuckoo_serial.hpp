#ifndef CUCKOO_SERIAL
#define CUCKOO_SERIAL

#include<iostream>
#include<stdlib.h>
#include<math.h>
#include<random>
#include"util.h"
#include<vector>


#define DIVISOR_BOUND 50    // Divisor should be in range 0-50

std::random_device rd;
std::mt19937 rnd(rd());


/*  
    For simplicity, we only consider storing keys 
    in the hash table and ignore any associated data.
*/

class cuckoo_serial
{

    
private:
    const uint _size;         // Size of the hash table
    uint _num_func;           // Number of hash functions
    entry* _table;         // Data
    para *_hash_func_param; // Store all the divisors. As the form of hash function we use is:  ((a*key+b) mod prime) mod size

    uint _eviction_bound;     // When the chain's length reaches evication bound, rebuid the hash table
    uint32_t _prime = 230038579;

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
    void gen_hash_funcs() {

        // Calculate bit width of value range and table size.
        int val_width = 8 * sizeof(uint32_t) - ceil(log2((double) _num_func));
        int size_width = ceil(log2((double) _size));

        // Generate randomized configurations.
        for (int i = 1; i <= _num_func; ++i) {     // At index 0 is a dummy function.
            if (val_width <= size_width)
                _hash_func_param[i] = {rand(), 0};
            else
                _hash_func_param[i] = {rand(), rand() % (val_width - size_width + 1)};
        }
    }
 

public:

    cuckoo_serial(const uint size,uint num_func);
    ~cuckoo_serial();

    void insert(uint32_t key);
    bool remove(uint32_t key);
    bool look_up(uint32_t key);
    void show_table();
};

cuckoo_serial::cuckoo_serial(const uint size, uint num_func): _size(size), _num_func(num_func)
{
    _table = new entry[size](); 
    _hash_func_param = new para[num_func]();
    //gen_hash_funcs();
    gen_hash_divisor();
    _eviction_bound = 4 * ceil(log2(size));
}


/* 
    @brief Destructor of the table
*/
cuckoo_serial::~cuckoo_serial()
{
    delete[] _table;
    delete[] _hash_func_param;
}



//TODO: implement insert
void cuckoo_serial::insert(uint32_t key)
{

    if (look_up(key)) 
    {
        //std::cout << "exist!" << std::endl;
        return;   
    }

    uint hash_index = 0;
    para param = _hash_func_param[hash_index];
    uint data_index = hash_func(key,param);
    

    uint chain_count = 0;
    while (chain_count < _eviction_bound)
    {
        // for (int i = 0 ; i < _num_func; i++)
        // {
        //     para param = _hash_func_param[i];
        //     uint data_index = hash_func(key, param);
        //     if (_table[data_index].data == 0 )
        //     {
        //         _table[data_index].data = key;
        //         _table[data_index].cur_hash_index = i;
        //         return;
        //     }
        // }
        if (_table[data_index].data == 0)
        {
            _table[data_index].data = key;
            _table[data_index].cur_hash_index = hash_index;
            return ;
        }
        
        else
        {

            uint32_t evict_key = _table[data_index].data;
            uint evict_hash_index = _table[data_index].cur_hash_index;


            _table[data_index].data = key;
            _table[data_index].cur_hash_index = hash_index;

            key = evict_key;
            hash_index = (evict_hash_index + 1) % _num_func;  

            param = _hash_func_param[hash_index];
            data_index = hash_func(key, param);
        }

        // para param = _hash_func_param[hash_index];
        // uint data_index = hash_func(key,param);

        // uint32_t evict_key = _table[data_index].data;
        // uint evict_hash_index = _table[data_index].cur_hash_index;

        // _table[data_index].data = key;
        // _table[data_index].cur_hash_index = hash_index;

        // key = evict_key;
        // hash_index = (evict_hash_index + 1) % _num_func;

        chain_count++;
    }

    std::cout << chain_count << std::endl;
    //rebuild_table(key);

}


/*  
    @brief removal of a key

    @retval true: Remove sucessfully
    @retval false: Given key is not in the table
*/
bool cuckoo_serial::remove(uint32_t key)
{
    for (uint i = 0; i < _num_func; i++)
    {
        int index = hash_func(key,_hash_func_param[i]);
        if (key == _table[index ].data)
        {
            _table[index].data = 0;
            _table[index].cur_hash_index = 0;
            return true;
        }
    }

    return false;
}


/*  
    @brief look up of a key to determine if it is in the table

    @retval true: Found
    @retval false: Not found
*/
bool cuckoo_serial::look_up(uint32_t key)
{
    for (uint i = 0; i < _num_func; i++)
    {
        int index = hash_func(key, _hash_func_param[i]);
        if (key == _table[index].data)
            return true;
    }

    return false;
}

void cuckoo_serial::show_table()
{
    //int count = 0;
    for (uint i = 0; i < _size; i++)
    {
        if(_table[i].data == 0 )
        {
            std::cout << "key: " << _table[i].data << " hash: " << _table[i].cur_hash_index << " index: " << i << std::endl;
        }
        
        // if( count > 1000000 &&_table[i].data != 0)
        // {
        //     std::cout << "key: " << _table[i].data << " index: " << i << std::endl;
        // }
        // if (count > 1000)
        //     break;
    }
}

void cuckoo_serial::rebuild_table(uint32_t key)
{
    //gen_hash_funcs();
    gen_hash_divisor();

    std::vector<uint32_t> temp_data;
    for (int i = 0; i < _size; i++)
    {
        if(_table[i].data != 0)
        {
            temp_data.push_back(_table[i].data);
            _table[i].data = 0;
            _table[i].cur_hash_index = 0;
        }

    }
    //show_table();
    temp_data.push_back(key);

    for (int i = 0; i < temp_data.size(); i++)
        insert(temp_data[i]);

}

#endif