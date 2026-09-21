# Problem Description

A manufacturer must cut rectangular items from a single large rectangular panel of known length and width. There are several item types, each defined by a length, a width, a profit, and a maximum number of available copies. All dimensions and profits are positive integers. The goal is to select and cut a subset of items from the panel so as to maximize the total profit of the items obtained.

All cuts must be guillotine cuts, meaning each cut is a straight line that runs parallel to one side of the rectangle being cut and extends completely from one edge to the opposite edge, splitting that rectangle into exactly two smaller rectangles. There is no limit on the number of sequential cutting stages, and cuts need not be restricted (that is, the position of a cut need not coincide with the boundary of any item). Cuts may be either horizontal or vertical.

The cutting process works as follows. The original panel is the initial plate. Any plate may be split by a single guillotine cut into two smaller plates. Each of those smaller plates may in turn be split again, and so on recursively. A plate whose dimensions exactly match those of some item type may instead be kept as a finished item rather than being cut further. Any plate that is too small to contain any item is discarded. The process continues until every remaining plate has either been kept as an item or discarded.

The manufacturer must decide how to recursively cut the original panel and which resulting item-sized plates to retain as finished items. The number of finished items of each type kept cannot exceed that type's maximum available copies. The quantity to maximize is the sum over all retained items of the item's profit times the number of copies retained.
