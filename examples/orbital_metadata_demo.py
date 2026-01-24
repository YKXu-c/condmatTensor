#!/usr/bin/env python3
"""Orbital Metadata System Demonstration.

This script demonstrates the new orbital metadata system for BaseTensor,
which provides structured orbital characterization as an alternative to
simple string-based orbital_names.

Features demonstrated:
1. Dictionary format usage
2. String format usage
3. OrbitalMetadata object usage
4. Backward compatibility (orbital_names only)
5. Metadata preservation through transformations
6. Helper query methods (get_f_orbitals, is_spinful_system, etc.)
"""

import torch
from condmatTensor.core import BaseTensor, OrbitalMetadata, get_device
from condmatTensor.lattice import BravaisLattice, HoppingModel, generate_k_path


def demo_string_format():
    """Demonstrate string format for orbital metadata."""
    print("=" * 60)
    print("Demo 1: String Format for Orbital Metadata")
    print("=" * 60)

    # Create orbital metadata from strings
    orbital_names = [
        "Ce1-f-spin_up-local-U7.0",
        "Ce1-f-spin_down-local-U7.0",
        "Cu-px-spin_up",
        "Cu-px-spin_down",
    ]

    # Create a simple k-space tensor
    N_k = 10
    n_orb = len(orbital_names)
    tensor = torch.zeros((N_k, n_orb, n_orb), dtype=torch.complex128)

    # Create BaseTensor with string orbital names
    H = BaseTensor(tensor=tensor, labels=["k", "orb_i", "orb_j"], orbital_names=orbital_names)

    print(f"Orbital names: {H.orbital_names}")
    print(f"Number of orbitals: {H.shape[-1]}")

    # Metadata is lazily generated from orbital_names
    print(f"\nMetadata (lazy generation):")
    for i, md in enumerate(H.orbital_metadatas):
        print(f"  {i}: {md}")

    # Test helper methods
    print(f"\nF-orbital indices: {H.get_f_orbitals()}")
    print(f"Spinful orbitals: {H.get_spinful_orbitals()}")
    print(f"Localized orbitals: {H.get_localized_orbitals()}")
    print(f"Is spinful system: {H.is_spinful_system()}")

    # Test query by site
    print(f"Ce1 orbitals: {H.get_orbitals_by_site('Ce1')}")
    print(f"Cu orbitals: {H.get_orbitals_by_site('Cu')}")

    print()


def demo_dict_format():
    """Demonstrate dictionary format for orbital metadata."""
    print("=" * 60)
    print("Demo 2: Dictionary Format for Orbital Metadata")
    print("=" * 60)

    # Create orbital metadata as dictionaries
    orbital_metadatas = [
        {"site": "La", "orb": "dxy", "spin": "up", "local": False, "U": None},
        {"site": "La", "orb": "dxy", "spin": "down", "local": False, "U": None},
        {"site": "O1", "orb": "px", "spin": "up", "local": True, "U": 5.0},
        {"site": "O1", "orb": "px", "spin": "down", "local": True, "U": 5.0},
    ]

    # Create a simple k-space tensor
    N_k = 10
    n_orb = len(orbital_metadatas)
    tensor = torch.zeros((N_k, n_orb, n_orb), dtype=torch.complex128)

    # Create BaseTensor with dictionary orbital metadata
    H = BaseTensor(
        tensor=tensor, labels=["k", "orb_i", "orb_j"], orbital_metadatas=orbital_metadatas
    )

    print(f"Orbital names (auto-generated): {H.orbital_names}")
    print(f"Number of orbitals: {H.shape[-1]}")

    # Metadata is explicitly set
    print(f"\nMetadata (from dictionaries):")
    for i, md in enumerate(H.orbital_metadatas):
        print(f"  {i}: site={md.site}, orb={md.orb}, spin={md.spin}, local={md.local}, U={md.U}")

    # Test helper methods
    print(f"\nF-orbital indices: {H.get_f_orbitals()}")
    print(f"Spinful orbitals: {H.get_spinful_orbitals()}")
    print(f"Localized orbitals: {H.get_localized_orbitals()}")
    print(f"Is spinful system: {H.is_spinful_system()}")

    # Test query by site
    print(f"La orbitals: {H.get_orbitals_by_site('La')}")
    print(f"O1 orbitals: {H.get_orbitals_by_site('O1')}")

    print()


def demo_orbital_metadata_objects():
    """Demonstrate OrbitalMetadata object usage."""
    print("=" * 60)
    print("Demo 3: OrbitalMetadata Objects")
    print("=" * 60)

    # Create OrbitalMetadata objects
    orbital_metadatas = [
        OrbitalMetadata(site="Yb", orb="f", spin="up", local=True, U=8.0),
        OrbitalMetadata(site="Yb", orb="f", spin="down", local=True, U=8.0),
        OrbitalMetadata(site="Pt", orb="dx2-y2", spin="up", local=False),
        OrbitalMetadata(site="Pt", orb="dx2-y2", spin="down", local=False),
    ]

    # Create a simple k-space tensor
    N_k = 10
    n_orb = len(orbital_metadatas)
    tensor = torch.zeros((N_k, n_orb, n_orb), dtype=torch.complex128)

    # Create BaseTensor with OrbitalMetadata objects
    H = BaseTensor(
        tensor=tensor, labels=["k", "orb_i", "orb_j"], orbital_metadatas=orbital_metadatas
    )

    print(f"Orbital names (auto-generated from to_string()):")
    for i, name in enumerate(H.orbital_names):
        print(f"  {i}: {name}")

    # Metadata is explicitly set
    print(f"\nMetadata:")
    for i, md in enumerate(H.orbital_metadatas):
        print(f"  {i}: {md.as_dict()}")

    # Test helper methods
    print(f"\nF-orbital indices: {H.get_f_orbitals()}")
    print(f"Spinful orbitals: {H.get_spinful_orbitals()}")
    print(f"Localized orbitals: {H.get_localized_orbitals()}")
    print(f"Is spinful system: {H.is_spinful_system()}")

    print()


def demo_backward_compatibility():
    """Demonstrate backward compatibility (orbital_names only)."""
    print("=" * 60)
    print("Demo 4: Backward Compatibility (orbital_names only)")
    print("=" * 60)

    # Create BaseTensor with only orbital_names (no metadata)
    orbital_names = ["orb_0", "orb_1", "orb_2", "orb_3"]

    N_k = 10
    n_orb = len(orbital_names)
    tensor = torch.zeros((N_k, n_orb, n_orb), dtype=torch.complex128)

    H = BaseTensor(tensor=tensor, labels=["k", "orb_i", "orb_j"], orbital_names=orbital_names)

    print(f"Orbital names: {H.orbital_names}")
    print(f"_orbital_metadatas initially: {H._orbital_metadatas}")

    # Metadata is lazily generated on first access
    print(f"\nAccessing orbital_metadatas property (lazy generation)...")
    metadatas = H.orbital_metadatas

    print(f"_orbital_metadatas after access: {H._orbital_metadatas}")
    print(f"\nLazily generated metadata:")
    for i, md in enumerate(metadatas):
        print(f"  {i}: {md}")

    # Helper methods still work (with defaults)
    print(f"\nF-orbital indices: {H.get_f_orbitals()}")
    print(f"Spinful orbitals: {H.get_spinful_orbitals()}")
    print(f"Is spinful system: {H.is_spinful_system()}")

    print()


def demo_metadata_preservation():
    """Demonstrate metadata preservation through transformations."""
    print("=" * 60)
    print("Demo 5: Metadata Preservation Through Transformations")
    print("=" * 60)

    # Create BaseTensor with metadata
    orbital_metadatas = [
        OrbitalMetadata(site="U", orb="f", spin="up", local=True, U=6.0),
        OrbitalMetadata(site="U", orb="f", spin="down", local=True, U=6.0),
    ]

    # Create real-space tensor H(R)
    N_R = 3  # 3 unit cells
    n_orb = len(orbital_metadatas)
    tensor = torch.zeros((N_R, n_orb, n_orb), dtype=torch.complex128)
    displacements = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    H_R = BaseTensor(
        tensor=tensor,
        labels=["R", "orb_i", "orb_j"],
        orbital_metadatas=orbital_metadatas,
        displacements=displacements,
    )

    print(f"H(R) labels: {H_R.labels}")
    print(f"H(R) orbital_metadatas: {[md.as_dict() for md in H_R.orbital_metadatas]}")

    # Transform to k-space
    k_points = torch.tensor([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]])
    H_k = H_R.to_k_space(k_points)

    print(f"\nH(k) labels: {H_k.labels}")
    print(f"H(k) orbital_metadatas preserved: {[md.as_dict() for md in H_k.orbital_metadatas]}")

    # Move to device (if available)
    device = get_device()
    if device.type != "cpu":
        H_k_device = H_k.to(device)
        print(f"\nH_k on {device}: orbital_metadatas preserved: {H_k_device.orbital_metadatas is not None}")

    print()


def demo_kagome_with_metadata():
    """Demonstrate Kagome lattice with orbital metadata."""
    print("=" * 60)
    print("Demo 6: Kagome Lattice with Orbital Metadata")
    print("=" * 60)

    # Create Kagome lattice with metadata (2D for high_symmetry_points support)
    a = 1.0
    cell_vectors = torch.tensor([
        [a/2, a*3**0.5/2],
        [a/2, -a*3**0.5/2],
    ])

    basis_positions = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.5, 0.0]),
        torch.tensor([0.0, 0.5]),
    ]

    num_orbitals = [1, 1, 1]

    lattice = BravaisLattice(
        cell_vectors=cell_vectors,
        basis_positions=basis_positions,
        num_orbitals=num_orbitals,
    )

    # Create TightBindingModel with metadata (note: orbital_labels are auto-generated)
    orbital_metadatas = [
        OrbitalMetadata(site="A", orb="px"),
        OrbitalMetadata(site="B", orb="py"),
        OrbitalMetadata(site="C", orb="pz"),
    ]

    model = TightBindingModel(
        lattice=lattice,
        orbital_metadatas=orbital_metadatas,
    )

    # Add hoppings using the auto-generated orbital labels
    # Labels are generated from to_string() which includes all metadata
    orbital_labels = model.orbital_labels
    print(f"Auto-generated orbital labels: {orbital_labels}")
    print(f"Using indices: 0, 1, 2 for hoppings")

    # Add hoppings by index (simpler for this demo)
    model.add_hopping(0, 1, [0, 0], 1.0)  # A -> B
    model.add_hopping(1, 2, [0, 0], 1.0)  # B -> C
    model.add_hopping(2, 0, [0, 0], 1.0)  # C -> A

    # Build H(k)
    k_path, labels = generate_k_path(lattice, ["G", "K", "M", "G"], n_per_segment=50)
    Hk = model.build_Hk(k_path)

    print(f"Lattice: {lattice}")
    print(f"Orbital names: {model.orbital_labels}")
    print(f"H(k) shape: {Hk.shape}")
    print(f"H(k) orbital_metadatas: {[md.as_dict() for md in Hk.orbital_metadatas]}")

    # Query orbitals by site
    for site in ["A", "B", "C"]:
        print(f"Site {site} orbitals: {Hk.get_orbitals_by_site(site)}")

    print()


def demo_string_parsing():
    """Demonstrate string parsing for various formats."""
    print("=" * 60)
    print("Demo 7: String Parsing for Various Formats")
    print("=" * 60)

    test_strings = [
        "Ce1-f-spin_up-local-U7.0",
        "La-dxy-spin_up",
        "px",
        "up",
        "Yb-f-spin_down-local-U8.0",
        "Cu-dx2-y2-spin_up-conductive",
    ]

    for s in test_strings:
        md = OrbitalMetadata.from_string(s)
        print(f"Input: '{s}'")
        print(f"  Parsed: site={md.site!r}, orb={md.orb!r}, spin={md.spin!r}, local={md.local}, U={md.U}")
        print(f"  to_string(): '{md.to_string()}'")
        print()

    print()


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("Orbital Metadata System Demonstration")
    print("=" * 60 + "\n")

    demo_string_format()
    demo_dict_format()
    demo_orbital_metadata_objects()
    demo_backward_compatibility()
    demo_metadata_preservation()
    demo_kagome_with_metadata()
    demo_string_parsing()

    print("=" * 60)
    print("All demonstrations completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
