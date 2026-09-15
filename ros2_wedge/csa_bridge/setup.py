from setuptools import setup

package_name = "csa_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="CSA project",
    maintainer_email="noreply@example.com",
    description=(
        "Drop-in CSA compression for one bandwidth-constrained pose topic -- "
        "batches and compresses geometry_msgs/PoseStamped via the real C++ "
        "codec, for the link where MCAP's own default zstd measurably loses."
    ),
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "compressor_node = csa_bridge.compressor_node:main",
            "decompressor_node = csa_bridge.decompressor_node:main",
        ],
    },
)
