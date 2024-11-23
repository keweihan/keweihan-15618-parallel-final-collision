# Collision Stress Test
Basic collision stress scene.

## Getting Started (GHC Linux)
Note: GHC has cmake, python preinstalled and in PATH. 

### Install dependencies
1. Run installer helper `python3 scripts/build.py install`

### Build and run
1. Build executable `python3 scripts/build.py build --type release`
    - Use `debug` for debuggable version (i.e. can use lldb.)
2. Execute runnable with `./build/bin/collisionStress`

## Getting Started (Mac/Windows)

1. install `python` (>=3.8)
2. [conan ](https://conan.io/downloads) 
    - ex. `pip install conan`
3. [cmake](https://cmake.org/download/) 
    - Ubuntu: `sudo apt install -y cmake`
4. Run installer helper `python ./scripts/build.py install`

### Build and run
1. Configure scene in `demos/collisionStress.cpp`. E.g. run headless (no window/rendering) or with window 
2. Build executable `python src/build.py build --type release`
    - Use `debug` for debuggable version. 
3. Execute runnable in `./build/bin/collisionStress`

## Notes

### Collision Logic Overview
#### Data Structures

**General entity-component system overview:**
- An `Entity` can have multiple `Component`. 
- The type of `Component` an entity has dictates behavior of the entity. 
- The behavior a Component adds to an entity is implemented in `Component::update()` and `Component::onCollide()` functions. 

**Key components for collision:**

`Collider` is the component for **bounds** information for a given entity (i.e. how wide is it, where can it collide) and gives an object the ability to collide.
- `include/Collider.h`
- `include/BoxCollider.h`


`PhysicsBody` defines the container of **physics** information for a given entity as well as defines logic for velocity updates in `onCollide()`. Requires Collider for collision logic.
- `include/PhysicsBody.h`

#### Collision Algorithm
Collision data structures have the following hierarchy
```
ColliderSystem --has one-->  ColliderGrid
ColliderGrid -has many-> ColliderCell
ColliderCell --has many-> Entities (Balls)
Ball (Entity) --has one -->  Components (BoxCollider, PhysicsBody)
```

Relevant files:
- `src/Collision/ColliderSystem.h`
- `src/Collision/ColliderCell.h`
- `src/Collision/ColliderGrid.h`


#### Other
Output for statistics:
`src/Core/RenderStreategy/HeadlessRenderer.cpp`

### Deployment Bugs
Ensure no spaces in directory names when using conan. 