#ifndef CPPUTIL_SOFTSWITCH_SHARED_STORE_H
#define CPPUTIL_SOFTSWITCH_SHARED_STORE_H

#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <memory>
#include <unordered_map>

namespace cpputil {

template <typename T>
class SharedStore {
 public:
  virtual ~SharedStore() {
    clear();
  }

  using ConstructFunc = std::function<T*()>;
  using DataPtr = std::shared_ptr<T>;

  DataPtr get(const std::string& key, ConstructFunc construct, bool check_nullptr = false) {
    {
      // Read lock.
      boost::shared_lock<boost::shared_mutex> lock(mutex_);
      const auto iter = store_.find(key);
      if (iter != store_.end()) {
        return iter->second;
      }
    }

    // Only allows one thread to update store_.
    boost::upgrade_lock<boost::shared_mutex> lock(mutex_);
    boost::upgrade_to_unique_lock<boost::shared_mutex> unique_lock(lock);

    // Double-checked-locking.
    const auto iter = store_.find(key);
    if (iter != store_.end()) {
      return iter->second;
    }
    // Update store_.
    auto ptr = DataPtr(construct());
    if (check_nullptr && ptr == nullptr) {
        return nullptr;
    } else {
        store_[key] = ptr;
        return ptr;
    }
  }

  void clear() {
    boost::upgrade_lock<boost::shared_mutex> lock(mutex_);
    boost::upgrade_to_unique_lock<boost::shared_mutex> unique_lock(lock);
    store_.clear();
  }

 private:
  // Only allows one thread to update the store_.
  boost::shared_mutex mutex_;
  // Key: version_key -> Value: DataPtr
  // Guarded by mutex_.
  std::unordered_map<std::string, DataPtr> store_;
};

}  // namespace cpputil

#endif  // CPPUTIL_SOFTSWITCH_SHARED_STORE_H