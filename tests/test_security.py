import pytest
from unittest.mock import MagicMock
from tools.security import (
    validate_file,
    validate_uploaded_files,
    MAX_FILE_SIZE_MB,
    MAX_FILES_COUNT,
)


class MockBuffer:
    """Mock for buffer object with nbytes attribute."""
    def __init__(self, size):
        self.nbytes = size

class MockUploadedFile:
    """Mock for Streamlit's UploadedFile."""
    def __init__(self, name, content):
        self.name = name
        self._content = content
    
    def getbuffer(self):
        return MockBuffer(len(self._content))
    
    def getvalue(self):
        return self._content


def make_pdf_file(name, content=b"test"):
    """Helper to create a mock PDF file with proper header."""
    full_content = b'%PDF-1.4 ' + content
    return MockUploadedFile(name, full_content)


class TestValidateFile:
    """Tests for validate_file function."""

    def test_valid_pdf(self):
        """Test that a valid PDF passes validation."""
        file = make_pdf_file("test.pdf", b"test content")
        
        is_valid, error = validate_file(file)
        
        assert is_valid is True
        assert error == ""

    def test_file_too_large(self):
        """Test that oversized files are rejected."""
        # Create content larger than MAX_FILE_SIZE_MB (36MB)
        content = b'x' * (int(MAX_FILE_SIZE_MB * 1024 * 1024) + 1)
        file = MockUploadedFile("test.pdf", b'%PDF-1.4 ' + content)
        
        is_valid, error = validate_file(file)
        
        assert is_valid is False
        assert "tamaño máximo" in error

    def test_invalid_extension(self):
        """Test that non-PDF files are rejected."""
        file = MockUploadedFile("test.exe", b'%PDF-1.4 test content')
        
        is_valid, error = validate_file(file)
        
        assert is_valid is False
        assert "Solo se permiten" in error

    def test_path_traversal_attempt(self):
        """Test that path traversal attempts are blocked."""
        # Use .pdf extension but with path traversal - extension check runs first!
        file = MockUploadedFile("../../../test.pdf", b'%PDF-1.4 test')
        
        is_valid, error = validate_file(file)
        
        assert is_valid is False
        assert "path traversal" in error.lower() or "inválido" in error.lower()

    def test_invalid_pdf_header(self):
        """Test that files without PDF header are rejected."""
        file = MockUploadedFile("test.pdf", b'NOT A PDF')
        
        is_valid, error = validate_file(file)
        
        assert is_valid is False
        assert "no es un PDF válido" in error

    def test_hidden_file(self):
        """Test that hidden files (starting with .) are rejected."""
        file = MockUploadedFile(".hidden.pdf", b'%PDF-1.4 test content')
        
        is_valid, error = validate_file(file)
        
        assert is_valid is False
        assert "inválido" in error.lower()

    def test_none_file(self):
        """Test that None file is rejected."""
        is_valid, error = validate_file(None)
        
        assert is_valid is False
        assert "no proporcionado" in error


class TestValidateUploadedFiles:
    """Tests for validate_uploaded_files function."""

    def test_multiple_valid_files(self):
        """Test that multiple valid files pass validation."""
        files = [
            make_pdf_file("doc1.pdf", b"content1"),
            make_pdf_file("doc2.pdf", b"content2"),
        ]
        valid_files, errors = validate_uploaded_files(files)
        assert len(valid_files) == 2
        assert len(errors) == 0

    def test_max_files_limit(self):
        """Test that exceeding MAX_FILES_COUNT is handled."""
        files = [
            make_pdf_file(f"doc{i}.pdf", b'content')
            for i in range(MAX_FILES_COUNT + 3)
        ]
        
        valid_files, errors = validate_uploaded_files(files)
        
        # Should be limited to MAX_FILES_COUNT
        assert len(valid_files) <= MAX_FILES_COUNT
        assert "máximo" in errors[0].lower()

    def test_mixed_valid_invalid(self):
        """Test that valid files are returned even with invalid ones."""
        files = [
            make_pdf_file("valid.pdf", b"content"),
            MockUploadedFile("invalid.exe", b'NOT PDF'),
            make_pdf_file("another.pdf", b"content"),
        ]
        
        valid_files, errors = validate_uploaded_files(files)
        
        assert len(valid_files) == 2
        assert len(errors) == 1
