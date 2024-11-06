import React, { useState } from 'react';

const AudioUpload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [notes, setNotes] = useState([]);
  const [uploadStatus, setUploadStatus] = useState('');

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('audio', selectedFile);

    try {
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });
      if (response.ok) {
        const data = await response.json();
        setNotes(data.notes);  // Display the detected notes
        setUploadStatus('File uploaded successfully!');
      } else {
        setUploadStatus('File upload failed.');
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      setUploadStatus('Error uploading file.');
    }
  };

  return (
    <>
    <div style={styles.container}>
      <h2 style={styles.header}>Upload War Audio File</h2>
      <form onSubmit={handleSubmit} style={styles.form}>
        <input
          type="file"
          accept="audio/*"
          onChange={handleFileChange}
          style={styles.fileInput}
        />
        <button type="submit" style={styles.button}>
          Upload
        </button>
      </form>
      {uploadStatus && <p style={styles.status}>{uploadStatus}</p>}
     
    </div>
    <br></br>
    <div style={styles.container}> {notes.length > 0 && (
        <div style={styles.notesContainer}>
          <h3 style={styles.notesHeader}>Detected Notes</h3>
          <ul style={styles.notesList}>
            {notes.map((note, index) => (
              <li key={index} style={styles.noteItem}>{note}</li>
            ))}
          </ul>
          {/* <MusicStaff notes={notes} /> */}

        </div>
      )}</div>
    </>
  );
};

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: '20px',
    borderRadius: '10px',
    boxShadow: '0 4px 8px rgba(0, 0, 0, 0.2)',
    maxWidth: '400px',
    margin: 'auto',
    backgroundColor: '#f4f4f8',
  },
  header: {
    fontSize: '1.5em',
    marginBottom: '10px',
    color: '#333',
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    width: '100%',
  },
  fileInput: {
    padding: '8px',
    marginBottom: '10px',
    border: '1px solid #ddd',
    borderRadius: '4px',
    width: '100%',
    boxSizing: 'border-box',
  },
  button: {
    padding: '10px 20px',
    border: 'none',
    borderRadius: '4px',
    backgroundColor: '#007bff',
    color: '#fff',
    fontSize: '1em',
    cursor: 'pointer',
  },
  status: {
    marginTop: '10px',
    color: '#007bff',
    fontSize: '0.9em',
    textAlign: 'center',
  },
  notesContainer: {
    marginTop: '20px',
    width: '100%',
    textAlign: 'left',
  },
  notesHeader: {
    fontSize: '1.2em',
    marginBottom: '5px',
    color: '#333',
  },
  notesList: {
    paddingLeft: '20px',
  },
  noteItem: {
    fontSize: '1em',
    color: '#555',
  },
};

export default AudioUpload;
