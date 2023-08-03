# Environment Configuration

This directory houses scripts designed to install requisite dependencies within a Singularity environment. The file [`piston.def`](./piston.def) serves as the container definition.

Before you can build a container, you must obtain a functional link to the HDOCK docking tool from the tool's [official repository](http://huanglab.phys.hust.edu.cn/software/hdocklite/). Once you have this link, you should replace the content on line 14 of `piston.def` with it.

To construct a Singularity container, execute the following commands in your bash terminal:

```bash
chmod +x ./build_singularity.sh
./build_singularity.sh
```

Alternatively, you can reach out to Vitalii Stebliankin via email at vsteb002@fiu.edu to request a pre-configured .sif container.