/*
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Cristhian Alberto Gonzales Castillo <cristhian@blazingdb.com>
 *     Copyright 2019 Jean Pierre Huaroto <jeanpierre@blazingdb.com>
 *     Copyright 2019 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
 *     Copyright 2019 Christian Noboa Mardini <christian@blazingdb.com>
 */

#ifndef _BZ_DB_COMM_FR_NODE_H_
#define _BZ_DB_COMM_FR_NODE_H_

#include <vector>
#include <memory>
#include <string>

class Node {
    public:
        Node(int port, const std::string &ip);
        virtual ~Node();

        //const std::string getDataFile() const noexcept;
        //bool isEncrypted() const noexcept;

    private:
        //class Private;
        //const std::unique_ptr<Private> pimpl; // private implementation
};

#endif /* _BZ_DB_COMM_FR_NODE_H_ */
