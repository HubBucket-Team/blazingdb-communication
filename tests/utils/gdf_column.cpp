#include "tests/utils/gdf_column.h"
#include <memory>
#include <cstring>

namespace blazingdb {
namespace test {

    auto GdfColumnDeleter = [](gdf_column* column) {
        delete[] column->data;
        delete[] column->valid;
        delete[] column->col_name;
        delete column;
    };

    std::shared_ptr<gdf_column> build(std::string&& data,
                                      std::string&& valid,
                                      gdf_dtype dtype,
                                      gdf_size_type null_count,
                                      gdf_time_unit time_unit,
                                      std::string&& name) {
        gdf_column* column = new gdf_column;

        column->data = new char[data.size()];
        std::strcpy((char*)column->data, data.c_str());

        column->valid = new gdf_valid_type[valid.size()];
        std::strcpy((char*)column->valid, valid.c_str());

        column->size = data.size();

        column->dtype = dtype;

        column->null_count = null_count;

        column->dtype_info = gdf_dtype_extra_info { .time_unit = time_unit };

        column->col_name = new char[name.size()];
        std::strcpy((char*)column->col_name, name.c_str());

        return std::shared_ptr<gdf_column>(column, GdfColumnDeleter);
    }

    bool operator==(const gdf_column& lhs, const gdf_column& rhs) {
        if (lhs.data != rhs.data) {
            return false;
        }
        if (lhs.valid != rhs.valid) {
            return false;
        }
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
        if (lhs.col_name != rhs.col_name) {
            return false;
        }
        return true;
    }

} // namespace blazingdb
} // namespace test
