#!/bin/sh

GEM_BUILD_ROOT="build-deb"
GEM_BUILD_SRC="${GEM_BUILD_ROOT}/python-oq"

GEM_ALWAYS_YES=false

mksafedir () {
    local dname

    dname="$1"
    if [ "$GEM_ALWAYS_YES" != "true" -a -d "$dname" ]; then
        echo "$dname already exists"
        echo "press Enter to continue or CTRL+C to abort"
        read a
    fi
    rm -rf $dname
    mkdir -p $dname
}

usage () {
    local ret

    ret=$1

    echo
    echo "USAGE:"
    echo "    $0 [-B] build debian source package, if -B argument is present binary package is build too"
    echo
    exit $ret
}

#
#  MAIN

BUILD_BINARIES=0
#  args management
while [ $# -gt 0 ]; do
    case $1 in
        -B|--binaries)
            BUILD_BINARIES=1
            ;;
        -h|--help)
            usage 0
            break
            ;;
        *)
            usage 1
            break
            ;;
    esac
    shift
done

DPBP_FLAG=""
if [ "$BUILD_BINARIES" -eq 0 ]; then
    DPBP_FLAG="-S"
fi

mksafedir "$GEM_BUILD_ROOT"

mksafedir "$GEM_BUILD_SRC"

git archive HEAD | (cd "$GEM_BUILD_SRC" ; tar xv)

git submodule init
git submodule update

#  "submodule foreach" vars: $name, $path, $sha1 and $toplevel:
git submodule foreach "git archive HEAD | (cd \"\${toplevel}/${GEM_BUILD_SRC}/\$path\" ; tar xv ) "


cd "$GEM_BUILD_SRC"

# mods pre-packaging
mv LICENSE         openquake
mv README.txt      openquake/README
mv celeryconfig.py openquake
mv logging.cfg     openquake
mv openquake.cfg   openquake
mv bin/openquake   bin/noq
rm openquake/bin/oqpath.py 
rm -rf $(find demos -mindepth 1 -maxdepth 1 | grep -v 'demos/simple_fault_demo_hazard' | grep -v 'demos/event_based_hazard')
dpkg-buildpackage $DPBP_FLAG
cd -

# exit 0
#
# DEVEL
GEM_BUILD_PKG="${GEM_SRC_PKG}/pkg"
mksafedir "$GEM_BUILD_PKG"
GEM_BUILD_EXTR="${GEM_SRC_PKG}/extr"
mksafedir "$GEM_BUILD_EXTR"
set -x
cp  ${GEM_BUILD_ROOT}/python-noq_*.deb  $GEM_BUILD_PKG
cd "$GEM_BUILD_EXTR"
dpkg -x $GEM_BUILD_PKG/python-noq_*.deb .
dpkg -e $GEM_BUILD_PKG/python-noq_*.deb 
