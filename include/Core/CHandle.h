#pragma once

#include "SimpleECSAPI.h"
#include "Core/ComponentPool.h"
#include <cstdint>

namespace SimpleECS
{
	/**
	 * Handler wrapper class for components
	 * Provides stable pointer semantics for a given component type 
	 * and associated entity.
	 */
	template <typename T>
	class Handle {
	public:
		Handle();
		Handle(ComponentPool<T>* pool, uint32_t eid) : _pool(pool), _eid(eid) {}
		
		T& operator*() {
			return *_pool->getComponent(_eid);
		}

		// Overload arrow operator
		T* operator->() {
			return _pool->getComponent(_eid);
		}

		// Explicit conversion operator to bool
		explicit operator bool() const {
			return _eid != -1;
		}

	private:
		uint32_t _eid;
		ComponentPool<T>* _pool;
	};

	template<typename T>
	inline Handle<T>::Handle()
	{
		_pool = nullptr;
		_eid = -1;
	}
}