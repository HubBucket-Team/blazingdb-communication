#include "gdf_column_cpp.h"

namespace blazingdb {
namespace test {
/*
    gdf_column_cpp::gdf_column_cpp()
    { }

    gdf_column_cpp::gdf_column_cpp(const gdf_column_cpp& object)
    : column{object.column},
      allocated_size_data{object.allocated_size_data},
      allocated_size_valid{object.allocated_size_valid},
      column_name{object.column_name},
      is_ipc_column{object.is_ipc_column},
      column_token{object.column_token}
    { }

    //gdf_column_cpp& operator=(const gdf_column_cpp& object)
*/
    gdf_size_type gdf_column_cpp::size() {
        return column->size;
    }

    gdf_column* gdf_column_cpp::get_gdf_column() {
        return column;
    }

    std::string gdf_column_cpp::name() const {
        return column_name;
    }

    bool gdf_column_cpp::is_ipc() {
        return is_ipc_column;
    }

    column_token_t gdf_column_cpp::get_column_token() {
        return column_token;
    }

    void gdf_column_cpp::setColumn(gdf_column* value) {
        column = value;
    }

    void gdf_column_cpp::setSizeData(std::size_t value) {
        allocated_size_data = value;
    }

    void gdf_column_cpp::setSizeValid(std::size_t value) {
        allocated_size_valid = value;
    }

    void gdf_column_cpp::setColumnName(std::string value) {
        column_name = value;
    }

    void gdf_column_cpp::setIPC(bool value) {
        is_ipc_column = value;
    }

    void gdf_column_cpp::setColumnToken(column_token_t value) {
        column_token = value;
    }

    gdf_column_cpp build(gdf_column* column,
                         std::size_t data_size,
                         std::size_t valid_size,
                         std::string column_name,
                         bool is_ipc,
                         column_token_t column_token) {
        gdf_column_cpp result;
        result.setColumn(column);
        result.setSizeData(data_size);
        result.setSizeValid(valid_size);
        result.setColumnName(column_name);
        result.setIPC(is_ipc);
        result.setColumnToken(column_token);
        return result;
    }

    bool operator==(const gdf_column_cpp& lhs, const gdf_column_cpp& rhs) {
        return false;
    }

} // namespace blazingdb
} // namespace test
