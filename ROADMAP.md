## A small roadmap records the updates


04/2025
- Update type annotations
- Move dataclasses for Pydantic
- Add a clustering module shrink down searching spaces


Two traffic modes are coupled (DC mode, PNR mode)...
- Use two classes: One for DC mode; one for PNR mode (one class for one ridesharing mode)
- Use a third manager class to combine multiple modes and solve potential conflicts among mode results


Consider to use a uniformed filters to filter out trips
- Both riders must provide a sequence of links to travel on
- The idea is to match the travelers


## Consider new interface to refactor the model
1. A trip cluster for a set of travelers. Only need to record the basic information of:
   1. Original travel path/distance for each traveler
2. A DC class to model DC shared trips
3. A PNR class to model PNR shared trips

