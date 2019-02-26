#include "tests/utils/gdf_column.h"
#include <random>
#include <memory>
#include <cstring>
#include <algorithm>

namespace blazingdb {
namespace test {

    auto CudfColumnDeleter = [](gdf_column* column) {
        delete[] column->data;
        delete[] column->valid;
        delete[] column->col_name;
        delete column;
    };

    std::size_t get_dtype_width(blazingdb::test::gdf_dtype dtype) {
        using blazingdb::test::gdf_dtype;

        switch (dtype) {
            case gdf_dtype::GDF_INT8:
                return 1;
            case gdf_dtype::GDF_INT16:
                return 2;
            case gdf_dtype::GDF_INT32:
            case gdf_dtype::GDF_FLOAT32:
                return 4;
            case gdf_dtype::GDF_INT64:
            case gdf_dtype::GDF_FLOAT64:
                return 8;
            default:
                return 0;
        }
    }

    gdf_column* clone(gdf_column* input_column) {
        gdf_column* result = new gdf_column;

        result->data = nullptr;
        if (input_column->data != nullptr) {
            result->data = new char[input_column->size];
            std::memcpy(result->data, input_column->data, input_column->size);
        }

        result->valid = nullptr;
        if (input_column->valid != nullptr) {
            std::size_t valid_size = input_column->size * get_dtype_width(input_column->dtype);
            result->valid = new gdf_valid_type[valid_size];
            std::memcpy(result->valid, input_column->valid, valid_size);
        }

        result->col_name = nullptr;
        if (input_column->col_name != nullptr) {
            result->col_name = new char[std::strlen(input_column->col_name)];
            std::strcpy(result->col_name, input_column->col_name);
        }

        result->size = input_column->size;
        result->dtype = input_column->dtype;
        result->null_count = input_column->null_count;
        result->dtype_info = input_column->dtype_info;

        return result;
    }

    gdf_column* make(char* input_data,
                     gdf_valid_type* input_valid,
                     std::size_t input_size,
                     gdf_dtype input_dtype,
                     std::string input_name) {
        gdf_column* result = new gdf_column;

        result->data = nullptr;
        if (input_data != nullptr) {
            result->data = new char[input_size];
            std::memcpy(result->data, input_data, input_size);
        }

        result->valid = nullptr;
        if (input_valid != nullptr) {
            std::size_t valid_size = input_size * get_dtype_width(input_dtype);
            result->valid = new gdf_valid_type[valid_size];
            std::memcpy(result->valid, input_valid, valid_size);
        }

        result->col_name = nullptr;
        if (!input_name.empty()) {
            result->col_name = new char[input_name.size()];
            std::strcpy(result->col_name, input_name.c_str());
        }

        result->size = input_size;
        result->dtype = input_dtype;

        return result;
    }

    std::shared_ptr<gdf_column> build(std::size_t size,
                                      gdf_dtype dtype,
                                      gdf_size_type null_count,
                                      gdf_time_unit time_unit,
                                      std::string&& name) {
        std::mt19937 rng;
        auto Generator = [&rng]() {
            return (rng() % 26) + 65;
        };

        gdf_column* column = new gdf_column;

        std::size_t data_size = size;
        column->data = new char[data_size];
        std::generate_n(column->data, data_size, Generator);

        std::size_t valid_size = size * get_dtype_width(dtype);
        column->valid = new gdf_valid_type[valid_size];
        std::generate_n((char*)column->valid, valid_size, Generator);

        column->col_name = new char[name.size()];
        std::strcpy(column->col_name, name.c_str());

        column->size = size;
        column->dtype = dtype;
        column->null_count = null_count;
        column->dtype_info = gdf_dtype_extra_info { .time_unit = time_unit };

        return std::shared_ptr<gdf_column>(column, CudfColumnDeleter);
    }

    bool operator==(const gdf_column& lhs, const gdf_column& rhs) {
        if (lhs.size != rhs.size) {
            return false;
        }
        if (lhs.dtype != rhs.dtype) {
            return false;
        }
        if (lhs.null_count != rhs.null_count) {
            return false;
        }
        if (lhs.dtype_info.time_unit != rhs.dtype_info.time_unit) {
            return false;
        }
        if ((lhs.col_name == nullptr) && (rhs.col_name != nullptr)) {
            return false;
        }
        if ((lhs.col_name != nullptr) && (rhs.col_name == nullptr)) {
            return false;
        }
        if ((lhs.col_name != nullptr) && (rhs.col_name != nullptr)) {
            if (strcmp(lhs.col_name, rhs.col_name) != 0) {
                return false;
            }
        }
        for (std::size_t k = 0; k < lhs.size; ++k) {
            if (lhs.data[k] != rhs.data[k]) {
                return false;
            }
        }
        for (std::size_t k = 0; k < (lhs.size * get_dtype_width(lhs.dtype)); ++k) {
            if (lhs.valid[k] != rhs.valid[k]) {
                return false;
            }
        }
        return true;
    }

} // namespace blazingdb
} // namespace test
