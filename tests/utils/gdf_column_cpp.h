#ifndef BLAZINGDB_COMMUNICATION_TEST_GDF_COLUMN_CPP_H
#define BLAZINGDB_COMMUNICATION_TEST_GDF_COLUMN_CPP_H

#include <string>
#include "tests/utils/gdf_column.h"

namespace blazingdb {
namespace test {

    typedef std::uint64_t column_token_t;

    class gdf_column_cpp {
    public:
        gdf_column_cpp() = default;

        gdf_column_cpp(gdf_column_cpp&& object);

        gdf_column_cpp(const gdf_column_cpp& object);

    public:
        gdf_column_cpp& operator=(gdf_column_cpp&&) = delete;

        gdf_column_cpp& operator=(const gdf_column_cpp&) = delete;

    public:
        ~gdf_column_cpp();

    public:
        gdf_size_type size() const;

        gdf_column* get_gdf_column() const;

        std::string name() const;

        bool is_ipc() const;

        column_token_t get_column_token() const;

        void set_column_token(column_token_t column_token);

    public:
        void create_gdf_column(gdf_dtype dtype,
                               std::size_t num_values,
                               void* input_data,
                               gdf_valid_type* input_valid,
                               std::size_t width_per_value,
                               const std::string& column_name);

        void create_gdf_column_for_ipc(gdf_dtype type, void * col_data,gdf_valid_type * valid_data, gdf_size_type num_values, gdf_size_type null_count, std::string column_name);

    private:
        void make(char* input_data,
                  gdf_valid_type* input_valid,
                  std::size_t data_size,
                  std::size_t valid_size,
                  bool is_ipc,
                  std::string column_name,
                  column_token_t column_token,
                  gdf_column* column);

    public:
        void setColumn(gdf_column* value);

        void setSizeData(std::size_t value);

        void setSizeValid(std::size_t value);

        void setColumnName(std::string value);

        void setIPC(bool value);

        void setColumnToken(column_token_t value);

    private:
        void deleteRalColumn();

    private:
        gdf_column* column{};
        std::size_t allocated_size_data{};
        std::size_t allocated_size_valid{};
        std::string column_name{};
        bool is_ipc_column{};
        column_token_t column_token{};

        friend bool operator==(const gdf_column_cpp& lhs, const gdf_column_cpp& rhs);
    };

    gdf_column_cpp build(gdf_column* column,
                         std::size_t data_size,
                         std::size_t valid_size,
                         std::string column_name,
                         bool is_ipc,
                         column_token_t column_token);

    bool operator==(const gdf_column_cpp& lhs, const gdf_column_cpp& rhs);

} // namespace blazingdb
} // namespace test

#endif //BLAZINGDB_COMMUNICATION_TEST_GDF_COLUMN_CPP_H
