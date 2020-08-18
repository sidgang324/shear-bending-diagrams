# Shear Force and Bending Moment Diagram Generator
Automatically generate shear force and bending moment diagrams.

## Usage
Use positive scalar values without units for beam length and support locations. Initialize the beam object as shown in `test.ipynb`.

The argument for the `add_loads()` method is a tuple of PointLoad and/or DistributedLoad objects.

The `draw_diagram()` method will output to the graph to the console.
