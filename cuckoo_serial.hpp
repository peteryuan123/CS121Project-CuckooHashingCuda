#ifndef CUCKOO_SERIAL
#define CUCKOO_SERIAL

#include<iostream>
#include<stdlib.h>
#include<math.h>

#define DIVISOR_BOUND 50    // Divisor should be in range 0-50


/*  
    For simplicity, we only consider storing keys 
    in the hash table and ignore any associated data.
*/

class cuckoo_serial
{
    
private:
    const uint size;         // Size of the hash table
    uint num_func;           // Number of hash functions
    uint32_t* table;         // Data
    uint *hash_func_divisor; // Store all the divisors. As the form of hash function we use is:  (key / divisor) % size
    uint eviction_bound;     // When the chain's length reaches evication bound, rebuid the hash table

private:

    void rebuild_table();
    /*  @brief General form of hash function
              It is easy to re-generate new hash functions when we reach eviction bound 
        @return
                index of the key
    */
    
    uint hash_func(uint32_t key, uint divisor)
    {
        return (key / divisor) % size ;
    }

    // Re-generate new divisior
    void gen_hash_divisor()
    {
        for (uint i = 0; i < num_func; i++)
            hash_func_divisor[i] = rand() % DIVISOR_BOUND;
    }
 

public:

    cuckoo_serial(const uint size,uint num_func);
    ~cuckoo_serial();

    bool insert(uint32_t key);
    bool remove(uint32_t key);
    bool look_up(uint32_t key);

};


cuckoo_serial::cuckoo_serial(const uint size, uint num_func): size(size), num_func(num_func)
{
    table = new uint32_t[size](); 
    eviction_bound = 4 * ceil(log2(size));
    
    hash_func_divisor = new uint[num_func];
    gen_hash_divisor();
}


/* 
    @brief Destructor of the table
*/
cuckoo_serial::~cuckoo_serial()
{
    delete[] table;
    delete[] hash_func_divisor;
}

//TODO: implement insert
bool cuckoo_serial::insert(uint32_t key)
{

}


/*  
    @brief removal of a key

    @retval true: Remove sucessfully
    @retval false: Given key is not in the table
*/
bool cuckoo_serial::remove(uint32_t key)
{
    for (uint i = 0; i < num_func; i++)
    {
        int index = hash_func(key,hash_func_divisor[i]);
        if (key == table[index])
        {
            table[index] = 0;
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
    for (uint i = 0; i < num_func; i++)
    {
        int index = hash_func(key,hash_func_divisor[i]);
        if (key == table[index])
            return true;
    }

    return false;
}




#endif