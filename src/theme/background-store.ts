const databaseName = 'vizflow';
const storeName = 'files';
const backgroundKey = 'background';

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(databaseName, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(storeName);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('indexedDB.open failed'));
  });
}

async function withStore<T>(mode: IDBTransactionMode, run: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await openDatabase();
  try {
    return await new Promise<T>((resolve, reject) => {
      const transaction = db.transaction(storeName, mode);
      const request = run(transaction.objectStore(storeName));
      transaction.oncomplete = () => resolve(request.result);
      transaction.onerror = () => reject(transaction.error ?? new Error('indexedDB transaction failed'));
      transaction.onabort = () => reject(transaction.error ?? new Error('indexedDB transaction aborted'));
    });
  } finally {
    db.close();
  }
}

export async function loadBackgroundBlob(): Promise<Blob | undefined> {
  const value: unknown = await withStore('readonly', (store) => store.get(backgroundKey));
  return value instanceof Blob ? value : undefined;
}

export async function saveBackgroundBlob(blob: Blob | undefined): Promise<void> {
  if (blob === undefined) await withStore('readwrite', (store) => store.delete(backgroundKey));
  else await withStore('readwrite', (store) => store.put(blob, backgroundKey));
}
