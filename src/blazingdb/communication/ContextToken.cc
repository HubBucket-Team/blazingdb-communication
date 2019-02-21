#include "ContextToken.h"

using namespace blazingdb::communication;

int ContextToken::TOKEN = 0;

ContextToken::ContextToken() : token_{TOKEN++} {
}

ContextToken::ContextToken(int token) : token_{token} {
}

bool ContextToken::operator==(const ContextToken& rhs) const {
  return token_ == rhs.token_;
}