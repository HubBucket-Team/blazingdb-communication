#ifndef BLAZINGDB_COMMUNICATION_TEST_CUSTRINGS_H
#define BLAZINGDB_COMMUNICATION_TEST_CUSTRINGS_H

struct NVStrings {};

struct NVCategory {
    static NVCategory* create_from_array(const char** strs, unsigned int count){
        return nullptr;
    }
    static NVCategory* create_from_strings(NVStrings& strs){
        return nullptr;
    }
};

#endif //BLAZINGDB_COMMUNICATION_TEST_CUSTRINGS_H
