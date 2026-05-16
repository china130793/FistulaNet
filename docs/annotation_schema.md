# Annotation schema

## Volumetric labels

- `primary_tract`: main fistula route from the internal opening toward the external opening or terminal extension.
- `secondary_tract`: branch tract connected to the primary route.
- `abscess_cavity`: fluid or inflammatory cavity connected or adjacent to the fistula system.
- `internal_opening`: tract entry point at the anal canal or rectal wall.
- `external_opening`: tract exit point near perianal skin when visible.
- `internal_anal_sphincter`: inner sphincter boundary used for anatomical coordinate learning.
- `external_anal_sphincter`: outer sphincter boundary used for crossing-depth quantification.
- `levator_region`: levator/puborectalis reference region used to flag high extension.
- `anal_canal_lumen`: lumen reference for clock-face and radial coordinate encoding.

## Topology labels

- `centerline`: skeletonized central route of the tract system.
- `branch_point`: node where a secondary tract diverges from the main path.
- `abscess_node`: graph node associated with a connected abscess cavity.
- `eas_crossing`: graph node or edge segment where the tract crosses the external anal sphincter.
- `horseshoe_arc`: curved posterior or circumferential extension when present.

## Graph edge types

- `primary_tract`
- `secondary_extension`
- `abscess_communication`
- `sphincter_crossing_segment`

## Biomarkers

- internal opening clock position
- branch burden
- abscess communication status
- external anal sphincter involvement percentage
- graph complexity index
