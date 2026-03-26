#include "c_api.h"

#include <string>

namespace {

struct AMErrorObject {
    am_status_code code;
    std::string message;
};

}// namespace

am_error_handle am_error_create(am_status_code code, const char* message) {
    try {
        AMErrorObject* error = new AMErrorObject();
        error->code = code;
        error->message = message == nullptr ? "" : message;
        return static_cast<am_error_handle>(error);
    } catch (...) {
        return nullptr;
    }
}

void am_error_destroy(am_error_handle error) {
    delete static_cast<AMErrorObject*>(error);
}

am_status_code am_error_code(am_error_handle error) {
    if (error == nullptr) {
        return AM_STATUS_UNKNOWN;
    }
    return static_cast<AMErrorObject*>(error)->code;
}

const char* am_error_message(am_error_handle error) {
    if (error == nullptr) {
        return "";
    }
    return static_cast<AMErrorObject*>(error)->message.c_str();
}
