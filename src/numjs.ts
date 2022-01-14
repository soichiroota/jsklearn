export function argsort(array: any[]): number[] {
  const arrayObject = array.map((value, idx) => {
    return { value, idx };
  });
  arrayObject.sort((a, b) => {
    if (a.value < b.value) {
      return -1;
    }
    if (a.value > b.value) {
      return 1;
    }
    return 0;
  });
  const argIndices = arrayObject.map((data) => data.idx);
  return argIndices;
}
