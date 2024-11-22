#pragma once
#pragma warning(disable: 4251) // ignores warning for dll warning. User does not use this class.

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include "Collision/Collider.h"
#include "boost/container/small_vector.hpp"

//#ifdef _TESTING
//	#ifdef SIMPLEECS_EXPORTS
//		#define SIMPLEECS_API __declspec(dllexport)
//	#else
//		#define SIMPLEECS_API __declspec(dllimport)
//	#endif
//#else
//	#define SIMPLEECS_API // Leave it blank in release mode
//#endif

#include "SimpleECSAPI.h"

namespace SimpleECS
{
	using ColliderCellIterator = boost::container::small_vector<Collider*, 5>::iterator;
	using ColliderConstCellIterator = boost::container::small_vector<Collider*, 5>::const_iterator;

	/**
	 * A list Collider storage structure. Wrapper for boost::small_vector to store colliders.
	 * Similar to boost::container::flat_set with boost::small_vector, but unsorted.
	 * 
	 * Best performing if less than 5 colliders stored.
	 * 
	 * Characteristics:
	 * - Contiguous and unordered storage of Collider pointers.
	 * - Small static stack storage.
	 * - Set semantics - colliders are unique.
	 * - Linear-time (O(n)) insertion to elements in container
	 * - Linear-time (O(n)) removal to elements in container
	 * 
	 */
	class SIMPLEECS_API ColliderCell
	{
	public:
		ColliderCell(const ColliderCell& other);
		ColliderCell(int defaultSize);
		ColliderCell();
		~ColliderCell();

		ColliderCell& operator=(const ColliderCell& other);

		ColliderCellIterator find(Collider* col);

		ColliderConstCellIterator begin() const;

		ColliderCellIterator begin();

		ColliderConstCellIterator end() const;

		ColliderCellIterator end();

		ColliderCellIterator erase(ColliderCellIterator o);

		ColliderCellIterator erase(Collider* col);

		Collider* back();

		size_t size();
		
		void insert(Collider* col);

	private:
		boost::container::small_vector<Collider*, 5> colList;
	};
}