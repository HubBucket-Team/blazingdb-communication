#include "NodeToken.h"

using namespace blazingdb::communication;

NodeToken::NodeToken(int token) : token_{token} {
}

bool NodeToken::operator==(const NodeToken& rhs) {
  return token_ == rhs.token_;
}