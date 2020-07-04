#ifndef UTIL_
#define UTIL_

#include<iostream>

std::random_device rd;
std::mt19937 rnd(rd());

struct para
{
    int a;
    int b;
};

struct entry
{
    uint32_t data;
    uint cur_hash_index;
};



#endif