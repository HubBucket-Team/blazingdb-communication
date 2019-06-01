#include "tests/utils/gdf_column_cpp.h"
#include "tests/utils/gdf_column.h"

namespace blazingdb {
namespace test {
    gdf_column_cpp::gdf_column_cpp(const gdf_column_cpp& object) {
        make(object.get_gdf_column()->data,
             object.get_gdf_column()->valid,
             object.allocated_size_data,
             object.allocated_size_valid,
             object.is_ipc_column,
             object.column_name,
             object.column_token,
             object.column);
    }

    gdf_column_cpp::gdf_column_cpp(gdf_column_cpp&& object) {
        column = object.column;
        allocated_size_data = object.allocated_size_data;
        allocated_size_valid = object.allocated_size_valid;
        column_name = object.column_name;
        is_ipc_column = object.is_ipc_column;
        column_token = object.column_token;

        object.column = nullptr;
    }

    gdf_column_cpp::~gdf_column_cpp() {
        deleteRalColumn();
    }

    gdf_size_type gdf_column_cpp::size() const {
        return column->size;
    }

    gdf_column* gdf_column_cpp::get_gdf_column() const {
        return column;
    }

    std::string gdf_column_cpp::name() const {
        return column_name;
    }

    bool gdf_column_cpp::is_ipc() const {
        return is_ipc_column;
    }

    column_token_t gdf_column_cpp::get_column_token() const {
        return column_token;
    }

    void gdf_column_cpp::set_column_token(column_token_t column_token) {
        this->column_token = column_token;
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

    void gdf_column_cpp::create_gdf_column(gdf_dtype dtype,
                                           std::size_t num_values,
                                           void* input_data,
                                           gdf_valid_type* input_valid,
                                           std::size_t width_per_value,
                                           const std::string& column_name) {
        column = blazingdb::test::make((char*)input_data, input_valid, num_values, dtype, column_name);
        allocated_size_valid = num_values * get_dtype_width(dtype);
        allocated_size_data = num_values;
        this->column_name = column_name;
        is_ipc_column = false;
        column_token = 0;
    }

    void gdf_column_cpp::create_gdf_column_for_ipc(gdf_dtype type, void * col_data,gdf_valid_type * valid_data, gdf_size_type num_values, gdf_size_type null_count, std::string column_name) {
        column = blazingdb::test::make((char*)input_data, input_valid, num_values, dtype, column_name);
        column->null_count = null_count;
        allocated_size_valid = num_values * get_dtype_width(dtype);
        allocated_size_data = num_values;
        this->column_name = column_name;
        is_ipc_column = true;
        column_token = 0;
    }

    void gdf_column_cpp::make(char* input_data,
              gdf_valid_type* input_valid,
              std::size_t input_data_size,
              std::size_t input_valid_size,
              bool input_ipc,
              std::string input_column_name,
              column_token_t input_column_token,
              gdf_column* input_column) {
        column = new gdf_column;

        column->data = new char[input_data_size];
        memcpy(column->data, input_data, input_data_size);

        column->valid = new gdf_valid_type[input_valid_size];
        memcpy(column->valid, input_valid, input_valid_size);

        column->col_name = new char[std::strlen(input_column->col_name)];
        std::strcpy(column->col_name, input_column->col_name);

        column->size = input_data_size;
        column->dtype = input_column->dtype;
        column->dtype_info = input_column->dtype_info;
        column->null_count = input_column->null_count;

        allocated_size_data = input_data_size;
        allocated_size_valid = input_valid_size;
        column_name = input_column_name;
        is_ipc_column = input_ipc;
        column_token = input_column_token;
    }

    void gdf_column_cpp::deleteRalColumn() {
        if (column != nullptr) {
            if (column->data != nullptr) {
                delete[] column->data;
            }
            if (column->valid != nullptr) {
                delete[] column->valid;
            }
            if (column->col_name != nullptr) {
                delete[] column->col_name;
            }
            delete column;
        }
    }

    gdf_column_cpp build(gdf_column* column,
                         std::size_t data_size,
                         std::size_t valid_size,
                         std::string column_name,
                         bool is_ipc,
                         column_token_t column_token) {
        gdf_column_cpp result;
        result.setColumn(clone(column));
        result.setSizeData(data_size);
        result.setSizeValid(valid_size);
        result.setColumnName(column_name);
        result.setIPC(is_ipc);
        result.setColumnToken(column_token);
        return result;
    }

    bool operator==(const gdf_column_cpp& lhs, const gdf_column_cpp& rhs) {
        if (lhs.size() != rhs.size()) {
            return false;
        }
        if (lhs.is_ipc() != rhs.is_ipc()) {
            return false;
        }
        if (lhs.name() != rhs.name()) {
            return false;
        }
        if (lhs.get_column_token() != rhs.get_column_token()) {
            return false;
        }
        return ((*lhs.get_gdf_column()) == (*rhs.get_gdf_column()));
    }

} // namespace blazingdb
} // namespace test
