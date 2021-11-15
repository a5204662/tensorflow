
#ifndef TENSORFLOW_CORE_UTIL_LRU_CACHE_H_
#define TENSORFLOW_CORE_UTIL_LRU_CACHE_H_

#include <list>
#include <memory>

#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

namespace lru {
/**
 * a base lock class that can be overridden
 */
class Lock {
 public:
  virtual ~Lock() {}
  virtual void lock() = 0;
  virtual void unlock() = 0;

 protected:
  Lock() {}

 private:
  Lock(const Lock&);
  const Lock& operator=(const Lock&);
};
/**
 *	Null or noop lock class that is the default synchronization
 *	used in lru cache.
 *
 *	a simple pthread based lock would look like this:
 *	<pre>
 *		class MutexLock : public Lock {
 public:
 MutexLock() { ::pthread_mutex_init(&m_lock, 0); }
 virtual ~MutexLock() { ::pthread_mutex_destroy(&m_lock); }
 void lock() { ::pthread_mutex_lock(&m_lock); }
 void unlock() { ::pthread_mutex_unlock(&m_lock); }
 private:
 pthread_mutex_t m_lock;
 }
 *	</pre>
 *
 */
class NullLock : public Lock {
 public:
  NullLock() {}
  virtual ~NullLock() {}
  void lock() {}
  void unlock() {}
};
/**
 *	helper class to auto lock and unlock within a scope
 */
class ScopedLock {
 public:
  ScopedLock(Lock& lock) : m_lock(lock) { m_lock.lock(); }
  virtual ~ScopedLock() { m_lock.unlock(); }

 private:
  Lock& m_lock;
  ScopedLock(const ScopedLock&);
  const ScopedLock& operator=(const ScopedLock&);
};
// a double linked list node
template <class Key, class Value>
struct Node {
  Node* prev;
  Node* next;
  Key key;
  Value value;
  Node(const Key& keyObj, const Value& valueObj)
      : prev(0), next(0), key(keyObj), value(valueObj) {}
  virtual ~Node() { cleanup(); }
  void cleanup() {
    if (next) {
      delete next;
    }
    next = 0;
    prev = 0;
  }
  void unlink() {
    if (next) {
      next->prev = prev;
    }
    if (prev) {
      prev->next = next;
    }
    next = 0;
    prev = 0;
  }
  template <class Visitor>
  void walk(Visitor& visitorFunc) {
    visitorFunc(*this);
    if (this->next) {
      this->next->walk(visitorFunc);
    }
  }
};

// a doubly linked list class
template <class Key, class Value>
struct List {
  Node<Key, Value>* head;
  Node<Key, Value>* tail;
  size_t size;
  List() : head(0), tail(0), size(0) {}
  virtual ~List() { clear(); }
  void clear() {
    if (head) {
      delete head;
    }
    head = 0;
    tail = 0;
    size = 0;
  }
  Node<Key, Value>* pop() {
    if (!head) {
      return 0;
    } else {
      Node<Key, Value>* newHead = head->next;
      head->unlink();
      Node<Key, Value>* oldHead = head;
      head = newHead;
      size--;
      if (size == 0) {
        tail = 0;
      }
      return oldHead;
    }
  }
  Node<Key, Value>* remove(Node<Key, Value>* node) {
    if (node == head) {
      head = node->next;
    }
    if (node == tail) {
      tail = node->prev;
    }
    node->unlink();
    size--;
    return node;
  }
  void push(Node<Key, Value>* node) {
    node->unlink();
    if (!head) {
      head = node;
    } else if (head == tail) {
      head->next = node;
      node->prev = head;
    } else {
      tail->next = node;
      node->prev = tail;
    }
    tail = node;
    size++;
  }
};

/**
 *	The LRU Cache class templated by
 *		Key - key type
 *		Value - value type
 *		MapType - an associative container like std::map
 *		LockType - a lock type derived from the Lock class (default:
 *NullLock = no synchronization)
 */
// template<class Key, class Value,
//		class MapType = std::map<Key, Node<Key, Value>*>,
//		class LockType = NullLock>

template <class Key, class Value, class MapFunc,
          class MapType = absl::flat_hash_map<Key, Node<Key, Value>*, MapFunc>,
          class LockType = NullLock>

class Cache {
 public:
  class KeyNotFound : public std::exception {
   public:
    const char* what() const throw() { return "KeyNotFound"; }
  };
  // -- methods
  Cache(const std::string& name = "", size_t maxSize = 64)
      : m_maxSize(maxSize), name_(name), fill_print_(false) {}
  virtual ~Cache() {}
  void clear() {
    ScopedLock scoped(m_lock);
    m_cache.clear();
    m_keys.clear();
  }
  bool insert(const Key& key, const Value& value, Value& del) {
    ScopedLock scoped(m_lock);
    typename MapType::iterator iter = m_cache.find(key);
    if (iter != m_cache.end()) {
      iter->second->value = value;
      m_keys.remove(iter->second);
      m_keys.push(iter->second);
    } else {
      Node<Key, Value>* n = new Node<Key, Value>(key, value);
      m_cache[key] = n;
      m_keys.push(n);
      PrintFillLog();
      del = prune();
      return false;
    }
    return true;
  }
  bool tryGet(const Key& key, Value& value) {
    ScopedLock scoped(m_lock);
    typename MapType::iterator iter = m_cache.find(key);
    if (iter == m_cache.end()) {
      return false;
    } else {
      m_keys.remove(iter->second);
      m_keys.push(iter->second);
      value = iter->second->value;
      return true;
    }
  }
  const Value& get(const Key& key) {
    ScopedLock scoped(m_lock);
    typename MapType::iterator iter = m_cache.find(key);
    if (iter == m_cache.end()) {
      //throw KeyNotFound();
    }
    m_keys.remove(iter->second);
    m_keys.push(iter->second);
    return iter->second->value;
  }
  void remove(const Key& key) {
    ScopedLock scoped(m_lock);
    typename MapType::iterator iter = m_cache.find(key);
    if (iter != m_cache.end()) {
      m_keys.remove(iter->second);
      m_cache.erase(iter);
    }
  }
  bool contains(const Key& key) {
    ScopedLock scoped(m_lock);
    return m_cache.find(key) != m_cache.end();
  }
  static void printVisitor(const Node<Key, Value>& node) {
    std::cout << "{" << node.key << ":" << node.value << "}" << std::endl;
  }

  void dumpDebug(std::ostream& os) const {
    ScopedLock scoped(m_lock);
    std::cout << "Cache Size : " << m_cache.size() << " (max:" << m_maxSize
              << ") " << std::endl;
    for (Node<Key, Value>* node = m_keys.head; node != NULL;
         node = node->next) {
      printVisitor(*node);
    }
  }

  void PrintFillLog() {
    if (!fill_print_ && size() == m_maxSize) {
      LOG(WARNING) << "LRUCache:" << name_ << " size :" << size() << " fill .";
      fill_print_ = true;
    }
  }

  size_t size() const {
    ScopedLock scoped(m_lock);
    return m_cache.size();
  }

 protected:
  Value prune() {
    if (m_maxSize > 0 && m_cache.size() >= m_maxSize) {
      Node<Key, Value>* n = m_keys.pop();
      m_cache.erase(n->key);
      Value v = n->value;
      delete n;
      return v;
    } else {
      return nullptr;
    }
  }

 protected:
  mutable LockType m_lock;
  MapType m_cache;
  List<Key, Value> m_keys;
  size_t m_maxSize;
  std::string name_;
  bool fill_print_;

 private:
  Cache(const Cache&);
  const Cache& operator=(const Cache&);
};
}  // namespace lru
}  // namespace tensorflow

#endif
