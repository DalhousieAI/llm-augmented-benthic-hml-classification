CONTEXT_BENTHICNET_SUB = (
    "You are a helpful assistant for understanding benthic environments, "
    "specializing in the classification of SEAFLOOR TERRAIN and SUBSTRATES."
)

CONTEXT_BENTHICNET_BIO = (
    "You are a helpful assistant for understanding benthic environments, "
    "specializing in the classification of BIOTA present on seafloor landscapes."
)

SUB_VARS = [
    'texture', 'color', 'depth', 'grain',
    'pattern', 'light', 'surface', 'sediment',
    'shape', 'clarity', 'coverage', 'composition',
    'algae', 'density', 'rock'
    ]

BIOTA_VARS = [
    'color', 'texture', 'density', 'shape',
    'pattern', 'distribution', 'depth', 'substrate',
    'size', 'type', 'structure', 'abundance',
    'light', 'coverage', 'growth_pattern',
    ]

SUB_HIERARCHY_GRAPH_TEXT = (
    "Substrate -> Unconsolidated (soft), Consolidated (hard)\n"
    "Unconsolidated (soft) -> Sand / mud (<2mm), Pebble / gravel\n"
    "Sand / mud (<2mm) -> Fine sand (no shell fragments), Coarse sand (with shell fragments), Mud / silt (<64um)\n"
    "Pebble / gravel -> Biologenic, Gravel (2-10mm), Pebble (10-64mm)\n"
    "Biologenic -> Coral rubble, Rhodoliths, Screwshells, Coquina / shellhash\n"
    "Consolidated (hard) -> Boulders, Rock, Cobbles\n"
    "Anthropogenic -> Deck, Tile\n"
)

BIO_HIERARCHY_GRAPH_TEXT = (
    "Biota -> Macroalgae, Sponges, Cnidaria, Fishes, Ascidians, Echinoderms, Molluscs, Bryozoa, Worms, Bacterial mats, Crustacea, Brachiopods, Microalgae, Seagrasses, Jellies, Sea spiders\n"
    "Macroalgae -> Erect fine branching, Filamentous / filiform, Encrusting, Articulated calcareous, Large canopy-forming, Laminate, Erect coarse branching, Sheet-like / membraneous, Globose / saccate\n"
    "Sponges -> Cup-likes, Erect forms, Crusts, Massive forms\n"
    "Cup-likes -> Cups, Tubes and chimneys, Barrels\n"
    "Cups -> Cup / goblet, Incomplete cup / curled fan\n"
    "Cup / goblet -> Thick, Smooth, Frilly, Irregular, Stalked, Thin\n"
    "Cnidaria -> Corals, Hydroids, Hydrocorals, True anemones, Colonial anemones, Tube anemones\n"
    "Corals -> Black & Octocorals, Stony corals\n"
    "Black & Octocorals -> Branching (3D), Fan (2D), Whip, Encrusting, Blue coral (Heliopora), Massive soft corals, Quill (seapen), Organ pipe coral (Tubipora)\n"
    "Branching (3D) -> Fleshy, Non-fleshy\n"
    "Fleshy -> Arborescent, Mushroom\n"
    "Erect forms -> Simple, Branching, Laminar, Palmate, Stalked\n"
    "Non-fleshy -> Bottle-brush, Arborescent, Bushy\n"
    "Bottle-brush -> Complex, Simple\n"
    "Erect fine branching -> Red, Brown, Green\n"
    "Crusts -> Encrusting, Creeping / ramose\n"
    "Encrusting -> Lumpy, Nodular, Oscula, Smooth, Granular, Spikey, Papillate, Thick, Fluffy, Rough\n"
    "Fan (2D) -> Fern-frond, Rigid\n"
    "Fern-frond -> Complex, Simple\n"
    "Fishes -> Bony fishes, Elasmobranchs, Eels\n"
    "Ascidians -> Stalked, Unstalked\n"
    "Stalked -> Solitary, Colonial\n"
    "Massive forms -> Simple, Balls, Cryptic\n"
    "Simple -> Lumpy, Irregular, Ball, Spikey, Ribbon, Oscula, Shapeless, Holey, Smooth, Creep, Globes, Laminar, Crust, Papillate, Rough, Nodular, Donut\n"
    "Branching -> Arborescent, Branching, Tubular, Fan\n"
    "Arborescent -> Irregular, Short, Thin, Digitate, Thorny, Stumpy\n"
    "Filamentous / filiform -> Turfing algae, Red, Brown, Green\n"
    "Turfing algae -> Red, Green\n"
    "Elasmobranchs -> Sharks, Rays & skates\n"
    "Bioturbation -> Dwelling traces, Feeding traces, Crawling traces, Organism tests, Waste casts\n"
    "Dwelling traces -> Single burrow, Crater cone, Vertical burrow, Elongate depression, Small mound\n"
    "Laminar -> Laminar, Fan\n"
    "Palmate -> Arborescent, Palmate, Pronghorn\n"
    "Hydroids -> 2D, Feathered, Bushy, Complex, Mossy\n"
    "Branching -> Pointed, Lumpy, Repent, Fine, Stumpy, Remose, Stubby, Thorny, Spindles, Thin, Digitate, Prostrate, Frilly, Short\n"
    "Creeping / ramose -> Creeping, Repent, Ramose, Tempura\n"
    "Crawling traces -> Thin trail\n"
    "Echinoderms -> Sea urchins, Sea stars, Sea cucumbers, Feather stars, Ophiuroids\n"
    "Sea urchins -> Regular urchins, Irregular urchins\n"
    "Unstalked -> Solitary, Colonial\n"
    "Molluscs -> Chitons, Bivalves, Gastropods, Cephalopods\n"
    "Encrusting -> Red, Brown, Green\n"
    "Red -> Calcareous, Non-calcareous\n"
    "Stony corals -> Encrusting, Foliose / plate, Branching, Massive, Staghorn, Digitate, Solitary, Bottlebrush, Sub-massive, Corymbose, Tabulate, Columnar\n"
    "Hydrocorals -> Submassive / encrusting\n"
    "True anemones -> Other anemones, Flytrap\n"
    "Bryozoa -> Soft, Hard\n"
    "Soft -> Foliaceous, Dendroid\n"
    "Articulated calcareous -> Red, Green\n"
    "Hard -> Massive, Fenestrate, Branching, Encrusting\n"
    "Large canopy-forming -> Brown\n"
    "Worms -> Acorn worms, Polychaetes, Penisworms, Flatworms\n"
    "Laminate -> Red, Brown, Green\n"
    "Erect coarse branching -> Red, Brown, Green\n"
    "Sheet-like / membraneous -> Brown, Green, Red\n"
    "Colonial anemones -> Zoanthids, Corallimorphs\n"
    "Bacterial mats -> Cyanobacteria\n"
    "Tubes and chimneys -> Tubular, Chimney\n"
    "Crustacea -> Barnacles, Crabs, Prawns / shrimps / mysids, Lobsters, Hermit crabs\n"
    "Barnacles -> Acorn\n"
    "Encrusting -> Lobate, Ridged / Folded\n"
    "Bivalves -> Scallop\n"
    "Crabs -> True crabs\n"
    "Prawns / shrimps / mysids -> Shrimps, Prawns\n"
    "Lobsters -> True lobsters, Squat lobsters, Rock lobsters\n"
    "Hermit crabs -> With anemones, With shell or stone home\n"
    "Sea cucumbers -> Benthic, Pelagic / swimming\n"
    "Polychaetes -> Tube worms, Other polychaetes\n"
    "Barrels -> Thick, Mouth, Tubular, Oscula, Donut, Rotund\n"
    "Seagrasses -> Strap-like leaves, Elliptical leaves\n"
    "Solitary -> Free-living, Mushroom (Fungiidae), Attached\n"
    "Jellies -> Siphonophores, Ctenophores, Salps, Hydromedusae\n"
    "Anthropogenic -> Benthic equipment, Fishing equipment, Trash, Cable, Tape\n"
    "Feather stars -> Unstalked crinoids\n"
    "Globose / saccate -> Green, Brown\n"
    "Balls -> Ball, Globular\n"
    "Ophiuroids -> Brittle / snake stars, Basket stars\n"
    "Organism tests -> Lumpy xenophyophore\n"
    "Waste casts -> Switchbacks, Spirals\n"
    "Cephalopods -> Squid, Octopods\n"
)

SUB_NODES = [
    "Anthropogenic",
    "Biologenic",
    "Boulders",
    "Coarse sand (with shell fragments)",
    "Cobbles",
    "Consolidated (hard)",
    "Coquina / shellhash",
    "Coral rubble",
    "Deck",
    "Fine sand (no shell fragments)",
    "Gravel (2-10mm)",
    "Mud / silt (<64um)",
    "Pebble (10-64mm)",
    "Pebble / gravel",
    "Rhodoliths",
    "Rock",
    "Sand / mud (<2mm)",
    "Screwshells",
    "Substrate",
    "Tile",
    "Unclear",
    "Unconsolidated (soft)",
    "Water column"
]

BIO_NODES = [
    '2D',
    'Acorn',
    'Acorn worms',
    'Anthropogenic',
    'Arborescent',
    'Articulated calcareous',
    'Ascidians',
    'Attached',
    'Bacterial mats',
    'Ball',
    'Balls',
    'Barnacles',
    'Barrels',
    'Basket stars',
    'Benthic',
    'Benthic equipment',
    'Biota',
    'Bioturbation',
    'Bivalves',
    'Black & Octocorals',
    'Blue coral (Heliopora)',
    'Bony fishes',
    'Bottle-brush',
    'Bottlebrush',
    'Brachiopods',
    'Branching',
    'Branching (3D)',
    'Brittle / snake stars',
    'Brown',
    'Bryozoa',
    'Bushy',
    'Cable',
    'Calcareous',
    'Cephalopods',
    'Chimney',
    'Chitons',
    'Cnidaria',
    'Colonial',
    'Colonial anemones',
    'Columnar',
    'Complex',
    'Corallimorphs',
    'Corals',
    'Corymbose',
    'Crabs',
    'Crater cone',
    'Crawling traces',
    'Creep',
    'Creeping',
    'Creeping / ramose',
    'Crust',
    'Crustacea',
    'Crusts',
    'Cryptic',
    'Ctenophores',
    'Cup / goblet',
    'Cup-likes',
    'Cups',
    'Cyanobacteria',
    'Dendroid',
    'Digitate',
    'Donut',
    'Dwelling traces',
    'Echinoderms',
    'Eels',
    'Elasmobranchs',
    'Elliptical leaves',
    'Elongate depression',
    'Encrusting',
    'Erect coarse branching',
    'Erect fine branching',
    'Erect forms',
    'Fan',
    'Fan (2D)',
    'Feather stars',
    'Feathered',
    'Feeding traces',
    'Fenestrate',
    'Fern-frond',
    'Filamentous / filiform',
    'Fine',
    'Fishes',
    'Fishing equipment',
    'Flatworms',
    'Fleshy',
    'Fluffy',
    'Flytrap',
    'Foliaceous',
    'Foliose / plate',
    'Free-living',
    'Frilly',
    'Gastropods',
    'Globes',
    'Globose / saccate',
    'Globular',
    'Granular',
    'Green',
    'Hard',
    'Hermit crabs',
    'Holey',
    'Hydrocorals',
    'Hydroids',
    'Hydromedusae',
    'Incomplete cup / curled fan',
    'Irregular',
    'Irregular urchins',
    'Jellies',
    'Laminar',
    'Laminate',
    'Large canopy-forming',
    'Lobate',
    'Lobsters',
    'Lumpy',
    'Lumpy xenophyophore',
    'Macroalgae',
    'Massive',
    'Massive forms',
    'Massive soft corals',
    'Microalgae',
    'Molluscs',
    'Mossy',
    'Mouth',
    'Mushroom',
    'Mushroom (Fungiidae)',
    'No visible biota',
    'Nodular',
    'Non-calcareous',
    'Non-fleshy',
    'Octopods',
    'Ophiuroids',
    'Organ pipe coral (Tubipora)',
    'Organism tests',
    'Oscula',
    'Other anemones',
    'Other polychaetes',
    'Palmate',
    'Papillate',
    'Pelagic / swimming',
    'Penisworms',
    'Pointed',
    'Polychaetes',
    'Prawns',
    'Prawns / shrimps / mysids',
    'Pronghorn',
    'Prostrate',
    'Quill (seapen)',
    'Ramose',
    'Rays & skates',
    'Red',
    'Regular urchins',
    'Remose',
    'Repent',
    'Ribbon',
    'Ridged / Folded',
    'Rigid',
    'Rock lobsters',
    'Rotund',
    'Rough',
    'Salps',
    'Scallop',
    'Sea cucumbers',
    'Sea spiders',
    'Sea stars',
    'Sea urchins',
    'Seagrasses',
    'Shapeless',
    'Sharks',
    'Sheet-like / membraneous',
    'Short',
    'Shrimps',
    'Simple',
    'Single burrow',
    'Siphonophores',
    'Small mound',
    'Smooth',
    'Soft',
    'Solitary',
    'Spikey',
    'Spindles',
    'Spirals',
    'Sponges',
    'Squat lobsters',
    'Squid',
    'Staghorn',
    'Stalked',
    'Stony corals',
    'Strap-like leaves',
    'Stubby',
    'Stumpy',
    'Sub-massive',
    'Submassive / encrusting',
    'Switchbacks',
    'Tabulate',
    'Tape',
    'Tempura',
    'Thick',
    'Thin',
    'Thin trail',
    'Thorny',
    'Trash',
    'True anemones',
    'True crabs',
    'True lobsters',
    'Tube anemones',
    'Tube worms',
    'Tubes and chimneys',
    'Tubular',
    'Turfing algae',
    'Unclear',
    'Unstalked',
    'Unstalked crinoids',
    'Vertical burrow',
    'Waste casts',
    'Whip',
    'With anemones',
    'With shell or stone home',
    'Worms',
    'Zoanthids'
]

SUB_KEYWORD_RULES = [
    ("anthropogenic", "Anthropogenic"),
    ("deck", "Anthropogenic > Deck"),
    ("tile", "Anthropogenic > Tile"),

    ("substrate", "Substrate"),
    ("hard substrate", "Substrate > Consolidated (hard)"),
    ("boulders", "Substrate > Consolidated (hard) > Boulders"),
    ("cobbles", "Substrate > Consolidated (hard) > Cobbles"),
    ("rock", "Substrate > Consolidated (hard) > Rock"),

    ("soft substrate", "Substrate > Unconsolidated (soft)"),
    ("pebble / gravel", "Substrate > Unconsolidated (soft) > Pebble / gravel"),
    ("biogenic gravel", "Substrate > Unconsolidated (soft) > Pebble / gravel > Biologenic"),
    ("coquina / shellhash", "Substrate > Unconsolidated (soft) > Pebble / gravel > Biologenic > Coquina / shellhash"),
    ("coral rubble", "Substrate > Unconsolidated (soft) > Pebble / gravel > Biologenic > Coral rubble"),
    ("rhodoliths", "Substrate > Unconsolidated (soft) > Pebble / gravel > Biologenic > Rhodoliths"),
    ("screwshells", "Substrate > Unconsolidated (soft) > Pebble / gravel > Biologenic > Screwshells"),
    ("gravel (2-10mm)", "Substrate > Unconsolidated (soft) > Pebble / gravel > Gravel (2-10mm)"),
    ("pebble (10-64mm)", "Substrate > Unconsolidated (soft) > Pebble / gravel > Pebble (10-64mm)"),

    ("sand / mud (<2mm)", "Substrate > Unconsolidated (soft) > Sand / mud (<2mm)"),
    ("coarse sand (shell fragments)", "Substrate > Unconsolidated (soft) > Sand / mud (<2mm) > Coarse sand (with shell fragments)"),
    ("fine sand (no shells)", "Substrate > Unconsolidated (soft) > Sand / mud (<2mm) > Fine sand (no shell fragments)"),
    ("mud / silt (<64um)", "Substrate > Unconsolidated (soft) > Sand / mud (<2mm) > Mud / silt (<64um)"),

    ("unclear", "Unclear"),
    ("water column", "Water column"),
]

BIO_KEYWORD_RULES = [
    ("anthropogenic", "Anthropogenic"),
    ("benthic equipment", "Anthropogenic > Benthic equipment"),
    ("cable", "Anthropogenic > Cable"),
    ("fishing equipment", "Anthropogenic > Fishing equipment"),
    ("tape", "Anthropogenic > Tape"),
    ("trash", "Anthropogenic > Trash"),

    ("biota", "Biota"),

    ("ascidians", "Biota > Ascidians"),
    ("stalked ascidians", "Biota > Ascidians > Stalked"),
    ("colonial stalked ascidians", "Biota > Ascidians > Stalked > Colonial"),
    ("solitary stalked ascidians", "Biota > Ascidians > Stalked > Solitary"),
    ("unstalked ascidians", "Biota > Ascidians > Unstalked"),
    ("colonial unstalked ascidians", "Biota > Ascidians > Unstalked > Colonial"),
    ("solitary unstalked ascidians", "Biota > Ascidians > Unstalked > Solitary"),

    ("bacterial mats", "Biota > Bacterial mats"),
    ("cyanobacterial mats", "Biota > Bacterial mats > Cyanobacteria"),

    ("brachiopods", "Biota > Brachiopods"),

    ("bryozoa", "Biota > Bryozoa"),
    ("hard bryozoa", "Biota > Bryozoa > Hard"),
    ("branching hard bryozoa", "Biota > Bryozoa > Hard > Branching"),
    ("encrusting hard bryozoa", "Biota > Bryozoa > Hard > Encrusting"),
    ("fenestrate hard bryozoa", "Biota > Bryozoa > Hard > Fenestrate"),
    ("massive hard bryozoa", "Biota > Bryozoa > Hard > Massive"),
    ("soft bryozoa", "Biota > Bryozoa > Soft"),
    ("dendroid soft bryozoa", "Biota > Bryozoa > Soft > Dendroid"),
    ("foliaceous soft bryozoa", "Biota > Bryozoa > Soft > Foliaceous"),

    ("cnidaria", "Biota > Cnidaria"),
    ("colonial anemones", "Biota > Cnidaria > Colonial anemones"),
    ("corallimorphs", "Biota > Cnidaria > Colonial anemones > Corallimorphs"),
    ("zoanthids", "Biota > Cnidaria > Colonial anemones > Zoanthids"),
    ("corals", "Biota > Cnidaria > Corals"),

    ("black & octocorals", "Biota > Cnidaria > Corals > Black & Octocorals"),
    ("blue coral (heliopora)", "Biota > Cnidaria > Corals > Black & Octocorals > Blue coral (Heliopora)"),
    ("branching (3d) black & octocorals", "Biota > Cnidaria > Corals > Black & Octocorals > Branching (3D)"),
    ("fleshy branching (3d) black & octocorals", "Biota > Cnidaria > Corals > Black & Octocorals > Branching (3D) > Fleshy"),
    ("arborescent fleshy branching (3d) black & octocorals", "Biota > Cnidaria > Corals > Black & Octocorals > Branching (3D) > Fleshy > Arborescent"),
    ("mushroom fleshy branching (3d) black & octocorals", "Biota > Cnidaria > Corals > Black & Octocorals > Branching (3D) > Fleshy > Mushroom"),
    ("non-fleshy branching (3d) black & octocorals", "Biota > Cnidaria > Corals > Black & Octocorals > Branching (3D) > Non-fleshy"),
    ("arborescent non-fleshy branching (3d) black & octocorals", "Biota > Cnidaria > Corals > Black & Octocorals > Branching (3D) > Non-fleshy > Arborescent"),
    ("bottle-brush non-fleshy branching (3d) black & octocorals", "Biota > Cnidaria > Corals > Black & Octocorals > Branching (3D) > Non-fleshy > Bottle-brush"),
    ("complex bottle-brush non-fleshy black & octocorals", "Biota > Cnidaria > Corals > Black & Octocorals > Branching (3D) > Non-fleshy > Bottle-brush > Complex"),
    ("simple bottle-brush non-fleshy black & octocorals", "Biota > Cnidaria > Corals > Black & Octocorals > Branching (3D) > Non-fleshy > Bottle-brush > Simple"),
    ("bushy non-fleshy branching (3d) black & octocorals", "Biota > Cnidaria > Corals > Black & Octocorals > Branching (3D) > Non-fleshy > Bushy"),
    ("encrusting black & octocorals", "Biota > Cnidaria > Corals > Black & Octocorals > Encrusting"),
    ("lobate encrusting black & octocorals", "Biota > Cnidaria > Corals > Black & Octocorals > Encrusting > Lobate"),
    ("ridged / folded encrusting black & octocorals", "Biota > Cnidaria > Corals > Black & Octocorals > Encrusting > Ridged / Folded"),
    ("fan (2d) black & octocorals", "Biota > Cnidaria > Corals > Black & Octocorals > Fan (2D)"),
    ("fern-frond fan (2d) black & octocorals", "Biota > Cnidaria > Corals > Black & Octocorals > Fan (2D) > Fern-frond"),
    ("complex fern-frond fan (2d) black & octocorals", "Biota > Cnidaria > Corals > Black & Octocorals > Fan (2D) > Fern-frond > Complex"),
    ("simple fern-frond fan (2d) black & octocorals", "Biota > Cnidaria > Corals > Black & Octocorals > Fan (2D) > Fern-frond > Simple"),
    ("rigid fan (2d) black & octocorals", "Biota > Cnidaria > Corals > Black & Octocorals > Fan (2D) > Rigid"),
    ("massive soft corals", "Biota > Cnidaria > Corals > Black & Octocorals > Massive soft corals"),
    ("organ pipe coral (tubipora)", "Biota > Cnidaria > Corals > Black & Octocorals > Organ pipe coral (Tubipora)"),
    ("quill (seapen) black & octocorals", "Biota > Cnidaria > Corals > Black & Octocorals > Quill (seapen)"),
    ("whip black & octocorals", "Biota > Cnidaria > Corals > Black & Octocorals > Whip"),

    ("stony corals", "Biota > Cnidaria > Corals > Stony corals"),
    ("bottlebrush stony corals", "Biota > Cnidaria > Corals > Stony corals > Bottlebrush"),
    ("branching stony corals", "Biota > Cnidaria > Corals > Stony corals > Branching"),
    ("columnar stony corals", "Biota > Cnidaria > Corals > Stony corals > Columnar"),
    ("corymbose stony corals", "Biota > Cnidaria > Corals > Stony corals > Corymbose"),
    ("digitate stony corals", "Biota > Cnidaria > Corals > Stony corals > Digitate"),
    ("encrusting stony corals", "Biota > Cnidaria > Corals > Stony corals > Encrusting"),
    ("foliose / plate stony corals", "Biota > Cnidaria > Corals > Stony corals > Foliose / plate"),
    ("massive stony corals", "Biota > Cnidaria > Corals > Stony corals > Massive"),
    ("solitary stony corals", "Biota > Cnidaria > Corals > Stony corals > Solitary"),
    ("attached solitary stony corals", "Biota > Cnidaria > Corals > Stony corals > Solitary > Attached"),
    ("free-living solitary stony corals", "Biota > Cnidaria > Corals > Stony corals > Solitary > Free-living"),
    ("mushroom (fungiidae)", "Biota > Cnidaria > Corals > Stony corals > Solitary > Mushroom (Fungiidae)"),
    ("staghorn stony corals", "Biota > Cnidaria > Corals > Stony corals > Staghorn"),
    ("sub-massive stony corals", "Biota > Cnidaria > Corals > Stony corals > Sub-massive"),
    ("tabulate stony corals", "Biota > Cnidaria > Corals > Stony corals > Tabulate"),

    ("hydrocorals", "Biota > Cnidaria > Hydrocorals"),
    ("submassive / encrusting hydrocorals", "Biota > Cnidaria > Hydrocorals > Submassive / encrusting"),

    ("hydroids", "Biota > Cnidaria > Hydroids"),
    ("2d hydroids", "Biota > Cnidaria > Hydroids > 2D"),
    ("bushy hydroids", "Biota > Cnidaria > Hydroids > Bushy"),
    ("complex hydroids", "Biota > Cnidaria > Hydroids > Complex"),
    ("feathered hydroids", "Biota > Cnidaria > Hydroids > Feathered"),
    ("mossy hydroids", "Biota > Cnidaria > Hydroids > Mossy"),

    ("true anemones", "Biota > Cnidaria > True anemones"),
    ("flytrap true anemones", "Biota > Cnidaria > True anemones > Flytrap"),
    ("other anemones", "Biota > Cnidaria > True anemones > Other anemones"),
    ("tube anemones", "Biota > Cnidaria > Tube anemones"),

    ("crustacea", "Biota > Crustacea"),
    ("barnacles", "Biota > Crustacea > Barnacles"),
    ("acorn barnacles", "Biota > Crustacea > Barnacles > Acorn"),

    ("crabs", "Biota > Crustacea > Crabs"),
    ("true crabs", "Biota > Crustacea > Crabs > True crabs"),

    ("hermit crabs", "Biota > Crustacea > Hermit crabs"),
    ("hermit crabs with anemones", "Biota > Crustacea > Hermit crabs > With anemones"),
    ("hermit crabs with shell or stone home", "Biota > Crustacea > Hermit crabs > With shell or stone home"),

    ("lobsters", "Biota > Crustacea > Lobsters"),
    ("rock lobsters", "Biota > Crustacea > Lobsters > Rock lobsters"),
    ("squat lobsters", "Biota > Crustacea > Lobsters > Squat lobsters"),
    ("true lobsters", "Biota > Crustacea > Lobsters > True lobsters"),

    ("prawns / shrimps / mysids", "Biota > Crustacea > Prawns / shrimps / mysids"),
    ("prawns", "Biota > Crustacea > Prawns / shrimps / mysids > Prawns"),
    ("shrimps", "Biota > Crustacea > Prawns / shrimps / mysids > Shrimps"),

    ("echinoderms", "Biota > Echinoderms"),
    ("feather stars", "Biota > Echinoderms > Feather stars"),
    ("unstalked crinoids", "Biota > Echinoderms > Feather stars > Unstalked crinoids"),

    ("ophiuroids", "Biota > Echinoderms > Ophiuroids"),
    ("basket stars", "Biota > Echinoderms > Ophiuroids > Basket stars"),
    ("brittle / snake stars", "Biota > Echinoderms > Ophiuroids > Brittle / snake stars"),

    ("sea cucumbers", "Biota > Echinoderms > Sea cucumbers"),
    ("benthic sea cucumbers", "Biota > Echinoderms > Sea cucumbers > Benthic"),
    ("pelagic / swimming sea cucumbers", "Biota > Echinoderms > Sea cucumbers > Pelagic / swimming"),

    ("sea stars", "Biota > Echinoderms > Sea stars"),

    ("sea urchins", "Biota > Echinoderms > Sea urchins"),
    ("irregular urchins", "Biota > Echinoderms > Sea urchins > Irregular urchins"),
    ("regular urchins", "Biota > Echinoderms > Sea urchins > Regular urchins"),

    ("fishes", "Biota > Fishes"),
    ("bony fishes", "Biota > Fishes > Bony fishes"),
    ("eels", "Biota > Fishes > Eels"),
    ("elasmobranchs", "Biota > Fishes > Elasmobranchs"),
    ("rays & skates", "Biota > Fishes > Elasmobranchs > Rays & skates"),
    ("sharks", "Biota > Fishes > Elasmobranchs > Sharks"),

    ("jellies", "Biota > Jellies"),
    ("ctenophores", "Biota > Jellies > Ctenophores"),
    ("hydromedusae", "Biota > Jellies > Hydromedusae"),
    ("salps", "Biota > Jellies > Salps"),
    ("siphonophores", "Biota > Jellies > Siphonophores"),

    ("macroalgae", "Biota > Macroalgae"),
    ("articulated calcareous macroalgae", "Biota > Macroalgae > Articulated calcareous"),
    ("green articulated calcareous macroalgae", "Biota > Macroalgae > Articulated calcareous > Green"),
    ("red articulated calcareous macroalgae", "Biota > Macroalgae > Articulated calcareous > Red"),

    ("encrusting macroalgae", "Biota > Macroalgae > Encrusting"),
    ("brown encrusting macroalgae", "Biota > Macroalgae > Encrusting > Brown"),
    ("green encrusting macroalgae", "Biota > Macroalgae > Encrusting > Green"),
    ("red encrusting macroalgae", "Biota > Macroalgae > Encrusting > Red"),
    ("calcareous red encrusting", "Biota > Macroalgae > Encrusting > Red > Calcareous"),
    ("non-calcareous red encrusting", "Biota > Macroalgae > Encrusting > Red > Non-calcareous"),

    ("erect coarse branching macroalgae", "Biota > Macroalgae > Erect coarse branching"),
    ("brown erect coarse branching macroalgae", "Biota > Macroalgae > Erect coarse branching > Brown"),
    ("green erect coarse branching macroalgae", "Biota > Macroalgae > Erect coarse branching > Green"),
    ("red erect coarse branching macroalgae", "Biota > Macroalgae > Erect coarse branching > Red"),

    ("erect fine branching macroalgae", "Biota > Macroalgae > Erect fine branching"),

    ("brown erect fine branching macroalgae", "Biota > Macroalgae > Erect fine branching > Brown"),
    ("green erect fine branching macroalgae", "Biota > Macroalgae > Erect fine branching > Green"),
    ("red erect fine branching macroalgae", "Biota > Macroalgae > Erect fine branching > Red"),

    ("filamentous / filiform macroalgae", "Biota > Macroalgae > Filamentous / filiform"),
    ("brown filamentous / filiform macroalgae", "Biota > Macroalgae > Filamentous / filiform > Brown"),
    ("green filamentous / filiform macroalgae", "Biota > Macroalgae > Filamentous / filiform > Green"),
    ("red filamentous / filiform macroalgae", "Biota > Macroalgae > Filamentous / filiform > Red"),

    ("turfing algae", "Biota > Macroalgae > Filamentous / filiform > Turfing algae"),
    ("green turfing algae", "Biota > Macroalgae > Filamentous / filiform > Turfing algae > Green"),
    ("red turfing algae", "Biota > Macroalgae > Filamentous / filiform > Turfing algae > Red"),

    ("globose / saccate macroalgae", "Biota > Macroalgae > Globose / saccate"),
    ("brown globose / saccate macroalgae", "Biota > Macroalgae > Globose / saccate > Brown"),
    ("green globose / saccate macroalgae", "Biota > Macroalgae > Globose / saccate > Green"),

    ("laminate macroalgae", "Biota > Macroalgae > Laminate"),
    ("brown laminate macroalgae", "Biota > Macroalgae > Laminate > Brown"),
    ("green laminate macroalgae", "Biota > Macroalgae > Laminate > Green"),
    ("red laminate macroalgae", "Biota > Macroalgae > Laminate > Red"),

    ("large canopy-forming macroalgae", "Biota > Macroalgae > Large canopy-forming"),
    ("brown large canopy-forming macroalgae", "Biota > Macroalgae > Large canopy-forming > Brown"),

    ("sheet-like / membraneous macroalgae", "Biota > Macroalgae > Sheet-like / membraneous"),
    ("brown sheet-like / membraneous macroalgae", "Biota > Macroalgae > Sheet-like / membraneous > Brown"),
    ("green sheet-like / membraneous macroalgae", "Biota > Macroalgae > Sheet-like / membraneous > Green"),
    ("red sheet-like / membraneous macroalgae", "Biota > Macroalgae > Sheet-like / membraneous > Red"),

    ("microalgae", "Biota > Microalgae"),

    ("molluscs", "Biota > Molluscs"),
    ("bivalves", "Biota > Molluscs > Bivalves"),
    ("scallops", "Biota > Molluscs > Bivalves > Scallop"),

    ("cephalopods", "Biota > Molluscs > Cephalopods"),
    ("octopods", "Biota > Molluscs > Cephalopods > Octopods"),
    ("squid", "Biota > Molluscs > Cephalopods > Squid"),

    ("chitons", "Biota > Molluscs > Chitons"),
    ("gastropods", "Biota > Molluscs > Gastropods"),

    ("sea spiders", "Biota > Sea spiders"),

    ("seagrasses", "Biota > Seagrasses"),
    ("elliptical-leaved seagrasses", "Biota > Seagrasses > Elliptical leaves"),
    ("strap-like seagrasses", "Biota > Seagrasses > Strap-like leaves"),

    ("sponges", "Biota > Sponges"),
    ("crustose sponges", "Biota > Sponges > Crusts"),

    ("creeping / ramose sponges", "Biota > Sponges > Crusts > Creeping / ramose"),
    ("creeping sponges", "Biota > Sponges > Crusts > Creeping / ramose > Creeping"),
    ("ramose sponges", "Biota > Sponges > Crusts > Creeping / ramose > Ramose"),
    ("repent sponges", "Biota > Sponges > Crusts > Creeping / ramose > Repent"),
    ("tempura sponges", "Biota > Sponges > Crusts > Creeping / ramose > Tempura"),

    ("encrusting sponges", "Biota > Sponges > Crusts > Encrusting"),
    ("fluffy encrusting sponges", "Biota > Sponges > Crusts > Encrusting > Fluffy"),
    ("granular encrusting sponges", "Biota > Sponges > Crusts > Encrusting > Granular"),
    ("lumpy encrusting sponges", "Biota > Sponges > Crusts > Encrusting > Lumpy"),
    ("nodular encrusting sponges", "Biota > Sponges > Crusts > Encrusting > Nodular"),
    ("oscula-bearing encrusting sponges", "Biota > Sponges > Crusts > Encrusting > Oscula"),
    ("papillate encrusting sponges", "Biota > Sponges > Crusts > Encrusting > Papillate"),
    ("rough encrusting sponges", "Biota > Sponges > Crusts > Encrusting > Rough"),
    ("smooth encrusting sponges", "Biota > Sponges > Crusts > Encrusting > Smooth"),
    ("spikey encrusting sponges", "Biota > Sponges > Crusts > Encrusting > Spikey"),
    ("thick encrusting sponges", "Biota > Sponges > Crusts > Encrusting > Thick"),

    ("cup-like sponges", "Biota > Sponges > Cup-likes"),

    ("barrel sponges", "Biota > Sponges > Cup-likes > Barrels"),
    ("donut barrel sponges", "Biota > Sponges > Cup-likes > Barrels > Donut"),
    ("mouth barrel sponges", "Biota > Sponges > Cup-likes > Barrels > Mouth"),
    ("oscula-bearing barrel sponges", "Biota > Sponges > Cup-likes > Barrels > Oscula"),
    ("rotund barrel sponges", "Biota > Sponges > Cup-likes > Barrels > Rotund"),
    ("thick barrel sponges", "Biota > Sponges > Cup-likes > Barrels > Thick"),
    ("tubular barrel sponges", "Biota > Sponges > Cup-likes > Barrels > Tubular"),

    ("cup sponges", "Biota > Sponges > Cup-likes > Cups"),
    ("cup / goblet sponges", "Biota > Sponges > Cup-likes > Cups > Cup / goblet"),
    ("frilly cup / goblet sponges", "Biota > Sponges > Cup-likes > Cups > Cup / goblet > Frilly"),
    ("irregular cup / goblet sponges", "Biota > Sponges > Cup-likes > Cups > Cup / goblet > Irregular"),
    ("smooth cup / goblet sponges", "Biota > Sponges > Cup-likes > Cups > Cup / goblet > Smooth"),
    ("stalked cup / goblet sponges", "Biota > Sponges > Cup-likes > Cups > Cup / goblet > Stalked"),
    ("thick cup / goblet sponges", "Biota > Sponges > Cup-likes > Cups > Cup / goblet > Thick"),
    ("thin cup / goblet sponges", "Biota > Sponges > Cup-likes > Cups > Cup / goblet > Thin"),

    ("incomplete cup / curled fan sponges", "Biota > Sponges > Cup-likes > Cups > Incomplete cup / curled fan"),

    ("tube and chimney sponges", "Biota > Sponges > Cup-likes > Tubes and chimneys"),
    ("chimney sponges", "Biota > Sponges > Cup-likes > Tubes and chimneys > Chimney"),
    ("tubular chimney sponges", "Biota > Sponges > Cup-likes > Tubes and chimneys > Tubular"),

    ("erect sponges", "Biota > Sponges > Erect forms"),
    ("branching erect sponges", "Biota > Sponges > Erect forms > Branching"),
    ("arborescent branching sponges", "Biota > Sponges > Erect forms > Branching > Arborescent"),
    ("digitate arborescent sponges", "Biota > Sponges > Erect forms > Branching > Arborescent > Digitate"),
    ("irregular arborescent sponges", "Biota > Sponges > Erect forms > Branching > Arborescent > Irregular"),
    ("short arborescent sponges", "Biota > Sponges > Erect forms > Branching > Arborescent > Short"),
    ("stumpy arborescent sponges", "Biota > Sponges > Erect forms > Branching > Arborescent > Stumpy"),
    ("thin arborescent sponges", "Biota > Sponges > Erect forms > Branching > Arborescent > Thin"),
    ("thorny arborescent sponges", "Biota > Sponges > Erect forms > Branching > Arborescent > Thorny"),

    ("branching sponges", "Biota > Sponges > Erect forms > Branching > Branching"),
    ("digitate branching sponges", "Biota > Sponges > Erect forms > Branching > Branching > Digitate"),
    ("fine branching sponges", "Biota > Sponges > Erect forms > Branching > Branching > Fine"),
    ("frilly branching sponges", "Biota > Sponges > Erect forms > Branching > Branching > Frilly"),
    ("lumpy branching sponges", "Biota > Sponges > Erect forms > Branching > Branching > Lumpy"),
    ("pointed branching sponges", "Biota > Sponges > Erect forms > Branching > Branching > Pointed"),
    ("prostrate branching sponges", "Biota > Sponges > Erect forms > Branching > Branching > Prostrate"),
    ("remose branching sponges", "Biota > Sponges > Erect forms > Branching > Branching > Remose"),
    ("repent branching sponges", "Biota > Sponges > Erect forms > Branching > Branching > Repent"),
    ("short branching sponges", "Biota > Sponges > Erect forms > Branching > Branching > Short"),
    ("spindle branching sponges", "Biota > Sponges > Erect forms > Branching > Branching > Spindles"),
    ("stubby branching sponges", "Biota > Sponges > Erect forms > Branching > Branching > Stubby"),
    ("stumpy branching sponges", "Biota > Sponges > Erect forms > Branching > Branching > Stumpy"),
    ("thin branching sponges", "Biota > Sponges > Erect forms > Branching > Branching > Thin"),
    ("thorny branching sponges", "Biota > Sponges > Erect forms > Branching > Branching > Thorny"),
    ("fan-shaped branching sponges", "Biota > Sponges > Erect forms > Branching > Fan"),
    ("tubular branching sponges", "Biota > Sponges > Erect forms > Branching > Tubular"),

    ("laminar sponges", "Biota > Sponges > Erect forms > Laminar"),

    ("fan-shaped laminar sponges", "Biota > Sponges > Erect forms > Laminar > Fan"),
    ("laminar sponges", "Biota > Sponges > Erect forms > Laminar > Laminar"),

    ("palmate sponges", "Biota > Sponges > Erect forms > Palmate"),
    ("arborescent palmate sponges", "Biota > Sponges > Erect forms > Palmate > Arborescent"),
    ("palmate-form sponges", "Biota > Sponges > Erect forms > Palmate > Palmate"),
    ("pronghorn palmate sponges", "Biota > Sponges > Erect forms > Palmate > Pronghorn"),

    ("simple erect sponges", "Biota > Sponges > Erect forms > Simple"),
    ("stalked erect sponges", "Biota > Sponges > Erect forms > Stalked"),

    ("massive sponges", "Biota > Sponges > Massive forms"),
    ("ball sponges", "Biota > Sponges > Massive forms > Balls"),
    ("spherical ball sponges", "Biota > Sponges > Massive forms > Balls > Ball"),
    ("globular ball sponges", "Biota > Sponges > Massive forms > Balls > Globular"),

    ("cryptic massive sponges", "Biota > Sponges > Massive forms > Cryptic"),
    ("simple massive sponges", "Biota > Sponges > Massive forms > Simple"),

    ("ball-shaped simple massive sponges", "Biota > Sponges > Massive forms > Simple > Ball"),
    ("creeping simple massive sponges", "Biota > Sponges > Massive forms > Simple > Creep"),
    ("crustose simple massive sponges", "Biota > Sponges > Massive forms > Simple > Crust"),
    ("donut-shaped simple massive sponges", "Biota > Sponges > Massive forms > Simple > Donut"),
    ("globose simple massive sponges", "Biota > Sponges > Massive forms > Simple > Globes"),
    ("holey simple massive sponges", "Biota > Sponges > Massive forms > Simple > Holey"),
    ("irregular simple massive sponges", "Biota > Sponges > Massive forms > Simple > Irregular"),
    ("laminar simple massive sponges", "Biota > Sponges > Massive forms > Simple > Laminar"),
    ("lumpy simple massive sponges", "Biota > Sponges > Massive forms > Simple > Lumpy"),
    ("nodular simple massive sponges", "Biota > Sponges > Massive forms > Simple > Nodular"),
    ("oscula-bearing simple massive sponges", "Biota > Sponges > Massive forms > Simple > Oscula"),
    ("papillate simple massive sponges", "Biota > Sponges > Massive forms > Simple > Papillate"),
    ("ribbon-like simple massive sponges", "Biota > Sponges > Massive forms > Simple > Ribbon"),
    ("rough simple massive sponges", "Biota > Sponges > Massive forms > Simple > Rough"),
    ("shapeless simple massive sponges", "Biota > Sponges > Massive forms > Simple > Shapeless"),
    ("smooth simple massive sponges", "Biota > Sponges > Massive forms > Simple > Smooth"),
    ("spikey simple massive sponges", "Biota > Sponges > Massive forms > Simple > Spikey"),

    ("worms", "Biota > Worms"),
    ("acorn worms", "Biota > Worms > Acorn worms"),
    ("flatworms", "Biota > Worms > Flatworms"),
    ("penisworms", "Biota > Worms > Penisworms"),
    ("polychaetes", "Biota > Worms > Polychaetes"),
    ("other polychaetes", "Biota > Worms > Polychaetes > Other polychaetes"),
    ("tube-dwelling polychaetes", "Biota > Worms > Polychaetes > Tube worms"),

    ("bioturbation", "Bioturbation"),

    ("crawling traces", "Bioturbation > Crawling traces"),
    ("thin crawling trails", "Bioturbation > Crawling traces > Thin trail"),

    ("dwelling traces", "Bioturbation > Dwelling traces"),
    ("crater-cone dwellings", "Bioturbation > Dwelling traces > Crater cone"),
    ("elongate depressions", "Bioturbation > Dwelling traces > Elongate depression"),
    ("single burrows", "Bioturbation > Dwelling traces > Single burrow"),
    ("small mounds", "Bioturbation > Dwelling traces > Small mound"),
    ("vertical burrows", "Bioturbation > Dwelling traces > Vertical burrow"),

    ("feeding traces", "Bioturbation > Feeding traces"),

    ("organism tests", "Bioturbation > Organism tests"),
    ("lumpy xenophyophore tests", "Bioturbation > Organism tests > Lumpy xenophyophore"),

    ("waste casts", "Bioturbation > Waste casts"),
    ("spiral waste casts", "Bioturbation > Waste casts > Spirals"),
    ("switchback waste casts", "Bioturbation > Waste casts > Switchbacks"),

    ("no visible biota", "No visible biota"),
    ("unclear", "Unclear"),
]

TAXONOMY_COLS = ['phylum', 'class', 'order', 'family', 'genus', 'species']
