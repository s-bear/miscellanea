#pragma once

#include <atomic>
#include <functional>

// This is free and unencumbered software released into the public domain.

/* Class to count copies of an object (or primitive type)
When the last copy of the item is deleted, calls release(item).
This is useful for adding reference counting to objects allocated by certain 
C libraries, e.g. OpenGL.
The counter is a std::atomic_size_t to provide some thread safety, but you still
need to be careful about passing references & pointers across thread boundaries.

Example:
//construct the copycount_t object with a lambda that calls glDeleteTextures
copycount_t<GLuint> tex_id( [](GLuint& id){ glDeleteTextures(1, &id); });
GLuint new_tex;
glGenTextures(1, &new_tex);
//check for errors etc
tex_id.set(std::move(new_tex)); //transfer ownership to tex_id
*/

template <typename T>
class copycount_t
{
public:
    typedef T item_type;
    typedef std::function<void(item_type &)> release_type;

protected:
    std::atomic_size_t *count;
    item_type item;
    release_type release;

public:
    //default constructor
    copycount_t() noexcept : count(nullptr) {}
    //set release function only -- use set(item_type) to set counted item later
    copycount_t(const release_type &release) noexcept : count(nullptr), release(release) {}
    copycount_t(release_type &&release) noexcept : count(nullptr), release(std::move(release)) {}
    //set item and release function together. Use move for item because copycount takes ownership of it
    copycount_t(item_type &&item, const release_type &release) : count(new std::atomic_size_t(1)), item(std::move(item)), release(release) {}
    copycount_t(item_type &&item, release_type &&release) : count(new std::atomic_size_t(1)), item(std::move(item)), release(release) {}
    //copy
    copycount_t(const copycount_t &other) : count(other.count), item(other.item), release(other.release)
    {
        //there's a chance that other could be destroyed (and call decrement & release) while we're copying
        //but it's not this class's responsibility to deal with that sort of thread safety issue
        increment();
    }
    //move
    copycount_t(copycount_t &&other) noexcept : count(other.count), item(std::move(other.item)), release(std::move(other.release))
    {
        //make sure to null other.count so that it doesn't decrement & release
        other.count = nullptr;
    }
    //destructor
    virtual ~copycount_t()
    {
        decrement();
    }
    //set: copy
    void set(const copycount_t &other)
    {
        if (&other != this)
        {
            //are we tracking the same counter?
            if (other.count != count)
            {
                decrement();
                count = other.count;
                increment();
                item = other.item;
            }
            //we can copy release even if the counters are the same -- allows for null situation
            release = other.release;
        }
    }
    copycount_t &operator=(const copycount_t &other)
    {
        set(other);
        return *this;
    }
    //set: move
    void set(copycount_t &&other)
    {
        if (&other != this)
        {
            if (other.count != count)
            {
                decrement();
                count = other.count();
                other.count() = nullptr;
                item = std::move(other.item);
            }
            //we can move release even if the counters are the same -- allows for the null counter situation
            release = std::move(other.release);
        }
    }
    copycount_t &operator=(copycount_t &&other)
    {
        set(std::move(other));
        return *this;
    }
    //set: move item (because we take ownership of it)
    void set(item_type &&item)
    {
        decrement();
        this->item = std::move(item);
        count = new std::atomic_size_t(1);
    }
    void set(item_type &&item, release_type &&release)
    {
        set(item);
        this->release = std::move(release);
    }
    void set(item_type &&item, const release_type &release)
    {
        set(item);
        this->release = release;
    }
    copycount_t &operator=(item_type &&item)
    {
        set(std::move(item));
        return *this;
    }
    //get() returns a const reference so that you can't modify the managed object
    //but you can make an unmanaged copy if you want to
    const item_type &get() const {
        if (count == nullptr) throw std::runtime_error("attempted access of invalid value");
        return item; 
    }
    operator item_type() const { return get(); }
    const release_type &get_release() const { return release; }

    bool valid() const noexcept {
        return count != nullptr;
    }
    //comparisons
    bool operator==(const copycount_t &other) const
    {
        return item == other.item && count == other.count;
    }
    bool operator==(const item_type &other) const
    {
        return item == other;
    }
    //other
protected:
    void increment()
    {
        if (count)
            ++(*count);
    }
    //decrease reference count & call release if necessary
    //protected because you shouldn't call this willy-nilly
    void decrement()
    {
        if (count && count->fetch_sub(1) == 1)
        {
            //we were the last holder of the item, release everything
            if (release)
                release(item);
            delete count;
            count = nullptr;
        }
    }
};