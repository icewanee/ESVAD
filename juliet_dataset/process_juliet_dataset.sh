gdown https://drive.google.com/uc?id=1kLAvOhEyMTnYRpPzs5T17yLBPZMLziVz && \
unzip -q 2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip && \
cp -r C/testcases testcases && cp -r C/testcasesupport testcasesupport && \
rm -rf C && \
rm 2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip && \
find testcases -type f \( -name "*.c" -o -name "*.cpp" -o -name "*.h" \) -exec sed -i 's/Winldap.h/winldap.h/g' {} + && \
python3.11 01_remove_unused_cwes.py && \
python3.11 02_emit_ll.py && \
python3.11 03_normalize_ll.py && \
python3.11 04_gen_cpg.py