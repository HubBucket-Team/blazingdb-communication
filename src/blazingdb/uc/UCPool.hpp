#pragma once 

#include <vector>

class UCPool {

public:
    static UCPool& getInstance()
    {
        static UCPool instance; 
        return instance;
    }
    void push(void * ptr) {
      pool.push_back(ptr);
    }
private:
    UCPool() {}                     

    UCPool(UCPool const&) = delete;             
    void operator=(UCPool const&) = delete;  

private: 
  std::vector<void *> pool; 
};